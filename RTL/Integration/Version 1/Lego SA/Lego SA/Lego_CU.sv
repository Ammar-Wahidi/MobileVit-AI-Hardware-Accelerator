// ================================================================
//  Lego_CU — Master Control Unit for the LEGO Systolic Array
//
//  A single FSM that drives all four L_SA_NxN_top tiles inside
//  Lego_SA simultaneously. No tile contains its own control unit;
//  this module is the sole source of load_w, valid_out, and busy.
//
//  Why a single shared CU:
//    When each tile had its own SA_CU, Lego_CU had to issue a
//    start pulse to each inner FSM. That pulse consumed one valid
//    cycle before the inner LOAD_W actually began, causing a
//    permanent phase offset that was impossible to compensate.
//    A single external CU drives load_w directly, eliminating
//    all inner FSMs and the offset they introduced.
//
//  State machine:
//    IDLE -> LOAD_W -> FEED_A -> DRAIN -> OUTPUT -> IDLE
//
//  Phase durations:
//    LOAD_W : N_TILE valid cycles.
//             load_w=1 on each valid cycle. All four tile PEs
//             latch weight_in simultaneously on every posedge.
//             valid_in=0 freezes this phase (counter holds).
//
//    FEED_A : N_TILE valid cycles.
//             load_w=0; PEs accumulate act_in * weight.
//             valid_in=0 freezes this phase (counter holds).
//             y_input_size is reserved for future extension but
//             FEED_A always runs exactly N_TILE cycles today.
//
//    DRAIN  : N_TILE-1 cycles, autonomous (ignores valid_in).
//             The pipeline drains; last partial sums propagate
//             through TRSDL so they are stable for OUTPUT tick 0.
//
//    OUTPUT : N_TILE cycles, autonomous (ignores valid_in).
//             valid_out=1; one de-skewed result row per cycle.
//             Caller captures psum_out on each of these cycles.
//
//  Stall behavior:
//    LOAD_W and FEED_A stall when valid_in=0.
//    DRAIN and OUTPUT never stall; once entered they run freely.
//
//  Protocol:
//    1. Assert valid_in=1.
//    2. Hold weight_in valid for N_TILE cycles (LOAD_W).
//    3. Hold act_in valid for N_TILE cycles (FEED_A).
//    4. Wait for valid_out=1 (DRAIN then OUTPUT automatic).
//    5. Capture psum_out for N_TILE cycles while valid_out=1.
//    6. Wait for busy=0 before next matmul.
//
//  Parameters:
//    N_TILE : tile dimension and phase cycle count (default 16)
//
// ================================================================

module Lego_CU #(
    parameter N_TILE = 16,
    parameter Y_INPUT_SIZE = 8
)(
    input  logic                        clk,
    input  logic                        rst_n,

    input  logic                        valid_in,      // data-valid signal and matmul start trigger
    input  logic [1:0]                  lego_type,     // shape selector (used by Lego_SA routing, not here)
    input  logic [Y_INPUT_SIZE-1:0]     y_input_size,  // reserved for future variable-length FEED_A

    output logic                        load_w,        // weight-latch enable, broadcast to all four tiles
    output logic                        valid_out,     // HIGH during OUTPUT phase (N_TILE cycles)
    output logic                        busy           // HIGH whenever state is not IDLE
);

// ── State encoding ────────────────────────────────────────────────
typedef enum logic [2:0] {
    IDLE   = 3'd0,
    LOAD_W = 3'd1,
    FEED_A = 3'd2,
    DRAIN  = 3'd3,
    OUTPUT = 3'd4
} state_t;

state_t state, next_state;

// ── Phase counter ─────────────────────────────────────────────────
// 8-bit counter; wide enough for any practical N_TILE.
localparam CNT_W = 8;

logic [CNT_W-1:0] cnt;
logic             cnt_en;   // 1 = increment counter this cycle
logic             cnt_rst;  // 1 = reset counter to zero this cycle

// ── State register ────────────────────────────────────────────────
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) state <= IDLE;
    else        state <= next_state;
end

// ── Counter register ─────────────────────────────────────────────
always_ff @(posedge clk or negedge rst_n) begin
    if      (!rst_n)  cnt <= '0;
    else if (cnt_rst) cnt <= '0;
    else if (cnt_en)  cnt <= cnt + 1'b1;
    // When neither flag is asserted, cnt holds its value (stall).
end

// ── FSM next-state and counter control ───────────────────────────
always_comb begin
    next_state = state;
    cnt_rst    = 1'b0;
    cnt_en     = 1'b0;

    case (state)

        // IDLE: wait for valid_in to trigger a new matmul.
        // Counter is kept cleared throughout IDLE.
        IDLE : begin
            cnt_rst = 1'b1;
            if (valid_in)
                next_state = LOAD_W;
        end

        // LOAD_W: load N_TILE weight rows into all four tiles.
        // Counter advances only when valid_in=1; freezes otherwise.
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

        // FEED_A: feed N_TILE activation rows through the tiles.
        // Counter advances only when valid_in=1; freezes otherwise.
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

        // DRAIN: pipeline drain, autonomous.
        // Runs N_TILE-1 cycles (exits when cnt reaches N_TILE-2).
        // The extra cycle ensures the last activation row's partial
        // sum settles through the final TRSDL stage before OUTPUT.
        DRAIN : begin
            cnt_en = 1'b1;
            if (cnt == N_TILE - 2) begin
                cnt_rst    = 1'b1;
                cnt_en     = 1'b0;
                next_state = OUTPUT;
            end
        end

        // OUTPUT: read results, autonomous.
        // valid_out=1 for N_TILE cycles; one result row per cycle.
        OUTPUT : begin
            cnt_en = 1'b1;
            if (cnt == N_TILE - 1) begin
                cnt_rst    = 1'b1;
                cnt_en     = 1'b0;
                next_state = IDLE;
            end
        end

        default : next_state = IDLE;

    endcase
end

// ── Output decode ─────────────────────────────────────────────────
// load_w is gated by valid_in so PEs only latch weight_in when
// the caller is presenting valid data on that cycle.
assign load_w    = (state == LOAD_W) && valid_in;
assign valid_out = (state == OUTPUT);
assign busy      = (state != IDLE);

endmodule