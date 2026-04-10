// ================================================================
//  SA_CU — Systolic Array Control Unit  (fully parameterized)
//
//  ── valid_in behavior ──────────────────────────────────────
//
//    valid_in = 1 : data is valid this cycle
//                   - In IDLE   : start a new matmul (enter LOAD_W)
//                   - In LOAD_W : accept weight_in, advance counter
//                   - In FEED_A : accept act_in,    advance counter
//
//    valid_in = 0 : data not valid this cycle
//                   - In IDLE   : stay IDLE
//                   - In LOAD_W : FREEZE — counter holds, load_w=0
//                                 this weight cycle is skipped/ignored
//                   - In FEED_A : FREEZE — counter holds
//                                 this activation cycle is skipped/ignored
//                   - In DRAIN  : ignored — DRAIN always runs autonomously
//                   - In OUTPUT : ignored — OUTPUT always runs autonomously
//
//  Effect: upstream can pause by deasserting valid_in mid-load or
//  mid-feed. The CU resumes exactly where it left off when valid_in
//  returns to 1. DRAIN and OUTPUT never stall once triggered.
//
//  ── State Machine ──────────────────────────────────────────
//
//    IDLE → LOAD_W → FEED_A → DRAIN → OUTPUT → IDLE
//
//  ── Phase durations ────────────────────────────────────────
//
//    LOAD_W : N_SIZE valid_in=1 cycles   load_w=1 only when valid_in=1
//    FEED_A : N_SIZE valid_in=1 cycles   act accepted only when valid_in=1
//    DRAIN  : N_SIZE-1 cycles            autonomous (no stall)
//    OUTPUT : N_SIZE   cycles            autonomous (no stall)
//
//  ── Ports ──────────────────────────────────────────────────
//
//    valid_in  : data-valid / start trigger
//    load_w    : HIGH when state==LOAD_W AND valid_in=1
//    valid_out : HIGH during OUTPUT phase
//    busy      : HIGH whenever CU is not IDLE
//
// ================================================================

module SA_CU #(
    parameter N_SIZE = 16
)(
    input  logic clk      ,
    input  logic rst_n    ,

    input  logic valid_in ,
    output logic load_w   ,
    output logic valid_out,
    output logic busy     
);

// ── State encoding ───────────────────────────────────────────
typedef enum logic [2:0] {
    IDLE   = 3'd0,
    LOAD_W = 3'd1,
    FEED_A = 3'd2,
    DRAIN  = 3'd3,
    OUTPUT = 3'd4
} state_t;

state_t state, next_state;

// ── Counter ──────────────────────────────────────────────────
localparam CNT_W = $clog2(N_SIZE) + 1;

logic [CNT_W-1:0] cnt;
logic              cnt_en;
logic              cnt_rst;

// ── State register ───────────────────────────────────────────
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) state <= IDLE;
    else        state <= next_state;
end

// ── Counter register ─────────────────────────────────────────
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)       cnt <= '0;
    else if (cnt_rst) cnt <= '0;
    else if (cnt_en)  cnt <= cnt + 1'b1;
    // else hold (stall)
end

// ── Next-state + control logic ───────────────────────────────
always_comb begin
    next_state = state;
    cnt_rst    = 1'b0;
    cnt_en     = 1'b0;

    case (state)

        // ── IDLE ─────────────────────────────────────────────
        IDLE : begin
            cnt_rst = 1'b1;              // keep counter at 0
            if (valid_in)
                next_state = LOAD_W;
        end

        // ── LOAD_W ───────────────────────────────────────────
        // Counter only advances when valid_in=1.
        // valid_in=0 → freeze here, load_w=0 (see output decode).
        LOAD_W : begin
            if (valid_in) begin
                cnt_en = 1'b1;
                if (cnt == N_SIZE - 1) begin
                    cnt_rst    = 1'b1;
                    cnt_en     = 1'b0;
                    next_state = FEED_A;
                end
            end
        end

        // ── FEED_A ───────────────────────────────────────────
        // Counter only advances when valid_in=1.
        // valid_in=0 → freeze here, upstream act_in ignored.
        FEED_A : begin
            if (valid_in) begin
                cnt_en = 1'b1;
                if (cnt == N_SIZE - 1) begin
                    cnt_rst    = 1'b1;
                    cnt_en     = 1'b0;
                    next_state = DRAIN;
                end
            end
        end

        // ── DRAIN ────────────────────────────────────────────
        // Autonomous — counts every cycle, ignores valid_in.
        DRAIN : begin
            cnt_en = 1'b1;
            if (cnt == N_SIZE - 2) begin
                cnt_rst    = 1'b1;
                cnt_en     = 1'b0;
                next_state = OUTPUT;
            end
        end

        // ── OUTPUT ───────────────────────────────────────────
        // Autonomous — counts every cycle, ignores valid_in.
        OUTPUT : begin
            cnt_en = 1'b1;
            if (cnt == N_SIZE - 1) begin
                cnt_rst    = 1'b1;
                cnt_en     = 1'b0;
                next_state = IDLE;
            end
        end

        default : next_state = IDLE;
    endcase
end

// ── Output decode ─────────────────────────────────────────────
// load_w gated by valid_in: if valid_in=0 in LOAD_W, PE must NOT
// latch weight_in (it would capture invalid data on that cycle).
assign load_w    = (state == LOAD_W) && valid_in;
assign valid_out = (state == OUTPUT);
assign busy      = (state != IDLE);

endmodule