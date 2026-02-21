// ================================================================
//  Lego_CU — Single Master Control Unit for the LEGO Systolic Array
//
//  This is the ONE AND ONLY control unit for the entire Lego_SA.
//  SA_NxN_top tiles no longer contain their own SA_CU — they are
//  pure datapaths that accept load_w directly from this module.
//
//  ── Why one CU is cleaner ──────────────────────────────────────
//
//  When each tile had its own SA_CU, Lego_CU had to drive
//  tile_valid_in as a proxy for starting the inner FSMs.  This
//  caused a permanent 1-cycle phase offset: the tile's start-pulse
//  consumed one valid cycle before its LOAD_W actually began,
//  making DRAIN/OUTPUT alignment impossible without workarounds.
//
//  With a single CU, Lego_CU drives load_w, valid_out, and busy
//  directly — no inner FSM, no offset, no proxy signals.
//
//  ── State machine ──────────────────────────────────────────────
//
//    IDLE → LOAD_W → FEED_A → DRAIN → OUTPUT → IDLE
//
//  ── Phase durations ────────────────────────────────────────────
//
//    LOAD_W : N_TILE cycles  (cnt 0 .. N_TILE-1)
//      load_w = 1 every cycle.
//      All 4 tiles latch weight_in on each posedge.
//      Caller must hold weight_in stable each cycle.
//
//    FEED_A : N_TILE cycles  (cnt 0 .. N_TILE-1)
//      load_w = 0, PEs accumulate act_in.
//      Caller must hold act_in stable each cycle.
//      Note: y_input_size port is kept for future extension but
//      currently FEED_A always runs N_TILE cycles (required by
//      the fixed-size SA_NxN datapath).
//
//    DRAIN  : N_TILE+1 cycles  (cnt 0 .. N_TILE)    autonomous
//      Pipeline drains; no inputs needed.
//
//    OUTPUT : N_TILE cycles    (cnt 0 .. N_TILE-1)  autonomous
//      valid_out = 1. Caller captures psum_out each cycle.
//
//  ── Stall behavior ─────────────────────────────────────────────
//
//    LOAD_W and FEED_A:
//      valid_in = 0 → counter freezes, no weight/act accepted.
//      valid_in = 1 → counter advances normally.
//      load_w follows valid_in in LOAD_W (PEs only latch when data valid).
//
//    DRAIN and OUTPUT:
//      Autonomous — run every cycle regardless of valid_in.
//      Once started, they cannot be stalled.
//
//  ── Protocol ───────────────────────────────────────────────────
//
//    1. Assert valid_in = 1.
//    2. During LOAD_W (N_TILE cycles): present weight_in each cycle.
//    3. During FEED_A (N_TILE cycles): present act_in each cycle.
//    4. DRAIN and OUTPUT run automatically.
//    5. Capture psum_out while valid_out = 1 (N_TILE cycles).
//    6. When busy = 0, system is ready for next matmul.
//
//  ── Outputs ────────────────────────────────────────────────────
//
//    load_w    : HIGH during LOAD_W AND valid_in=1
//                Drives all 4 tile SA_NxN datapaths directly.
//    valid_out : HIGH during OUTPUT phase (N_TILE cycles)
//    busy      : HIGH whenever state ≠ IDLE
//
// ================================================================

module Lego_CU #(
    parameter N_TILE = 16
)(
    input  logic       clk          ,
    input  logic       rst_n        ,

    input  logic       valid_in     ,    // data-valid / start trigger
    input  logic [1:0] lego_type    ,    // shape select (used by Lego_SA datapath)
    input  logic [7:0] y_input_size ,    // reserved — currently N_TILE always

    output logic       load_w       ,    // → all tile SA_NxN load_w ports
    output logic       valid_out    ,    // output phase flag
    output logic       busy              // HIGH when not IDLE
);

// ── State encoding ──────────────────────────────────────────────
typedef enum logic [2:0] {
    IDLE   = 3'd0,
    LOAD_W = 3'd1,
    FEED_A = 3'd2,
    DRAIN  = 3'd3,
    OUTPUT = 3'd4
} state_t;

state_t state, next_state;

// ── Counter ─────────────────────────────────────────────────────
localparam CNT_W = 8 ;

logic [CNT_W-1:0] cnt;
logic             cnt_en;
logic             cnt_rst;

// ── State register ──────────────────────────────────────────────
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) state <= IDLE;
    else        state <= next_state;
end

// ── Counter register ────────────────────────────────────────────
always_ff @(posedge clk or negedge rst_n) begin
    if      (!rst_n)  cnt <= '0;
    else if (cnt_rst) cnt <= '0;
    else if (cnt_en)  cnt <= cnt + 1'b1;
end

// ── FSM ─────────────────────────────────────────────────────────
always_comb begin
    next_state = state;
    cnt_rst    = 1'b0;
    cnt_en     = 1'b0;

    case (state)

        // ── IDLE ─────────────────────────────────────────────
        // Wait for valid_in. Counter held at 0.
        IDLE : begin
            cnt_rst = 1'b1;
            if (valid_in)
                next_state = LOAD_W;
        end

        // ── LOAD_W ───────────────────────────────────────────
        // N_TILE valid cycles.
        // load_w=1 drives all tile PEs to latch weight_in.
        // valid_in=0 freezes this phase (counter holds).
        LOAD_W : begin
            if (valid_in) begin
                cnt_en = 1'b1;
                if (cnt == N_TILE - 1) begin
                    cnt_rst    = 1'b1;
                    cnt_en     = 1'b0;
                    next_state = FEED_A;
                end
            end
        end

        // ── FEED_A ───────────────────────────────────────────
        // N_TILE valid cycles.
        // load_w=0, PEs accumulate act×weight each cycle.
        // valid_in=0 freezes this phase (counter holds).
        FEED_A : begin
            if (valid_in) begin
                cnt_en = 1'b1;
                if (cnt == N_TILE - 1) begin
                    cnt_rst    = 1'b1;
                    cnt_en     = 1'b0;
                    next_state = DRAIN;
                end
            end
        end

        // ── DRAIN ────────────────────────────────────────────
        // N_TILE cycles — autonomous, no stall.
        // Pipeline drains skewed partial sums through TRSDL.
        // Exits at cnt==N_TILE (runs N+1 ticks: cnt 0..N_TILE).
        // The extra cycle lets the last act row's contribution
        // settle through the final TRSDL register stage before
        // OUTPUT tick 0 samples psum_out.
        DRAIN : begin
            cnt_en = 1'b1;
            if (cnt == N_TILE -2) begin
                cnt_rst    = 1'b1;
                cnt_en     = 1'b0;
                next_state = OUTPUT;
            end
        end

        // ── OUTPUT ───────────────────────────────────────────
        // N_TILE cycles — autonomous, no stall.
        // valid_out=1. One de-skewed output row appears per cycle.
        OUTPUT : begin
            cnt_en = 1'b1;
            if (cnt == N_TILE -1) begin
                cnt_rst    = 1'b1;
                cnt_en     = 1'b0;
                next_state = IDLE;
            end
        end

        default : next_state = IDLE;

    endcase
end

// ── Output decode ────────────────────────────────────────────────
// load_w: gated by valid_in so PEs only latch when data is valid.
assign load_w    = (state == LOAD_W) && valid_in;
assign valid_out = (state == OUTPUT);
assign busy      = (state != IDLE);

endmodule