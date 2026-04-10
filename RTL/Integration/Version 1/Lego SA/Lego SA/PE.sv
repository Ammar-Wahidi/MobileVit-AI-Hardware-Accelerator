// ================================================================
//  PE — Processing Element
//
//  A single weight-stationary MAC cell in the systolic array.
//  Each PE holds one weight value and computes one multiply-
//  accumulate per cycle during the feed phase.
//
//  Weight loading:
//    When load_w=1 and transpose_en=0, the PE latches w_in_down
//    (weight travels upward through the column, row by row).
//    When load_w=1 and transpose_en=1, the PE latches w_in_left
//    (weight travels rightward through the row, column by column).
//
//  Accumulation:
//    When load_w=0, the PE multiplies in_act by W_reg and adds
//    in_psum, then registers the result as out_psum.
//    Activation is also registered and forwarded right (out_act)
//    so that the next PE in the row receives it one cycle later.
//
//  Weight propagation:
//    w_out_up    carries the incoming weight upward (normal mode).
//    w_out_right carries the incoming weight rightward (transpose mode).
//    Only one direction is active at a time; the other is zero.
//
//  Parameters:
//    DATA_W     : bit-width of activations and weights  (default 8)
//    DATA_W_OUT : bit-width of partial sum accumulator  (default 32)
//
// ================================================================

module PE #(
    parameter DATA_W     = 8,
    parameter DATA_W_OUT = 32
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // Activation path — flows left to right across a row
    input  logic [DATA_W-1:0]       in_act,          // activation from the left neighbor
    output logic [DATA_W-1:0]       out_act,         // activation forwarded to the right neighbor

    // Partial sum path — flows top to bottom down a column
    input  logic [DATA_W_OUT-1:0]   in_psum,         // partial sum from the PE above
    output logic [DATA_W_OUT-1:0]   out_psum,        // accumulated partial sum to the PE below

    // Weight load path — direction selected by transpose_en
    input  logic [DATA_W-1:0]       w_in_down,       // weight arriving from below  (normal mode)
    input  logic [DATA_W-1:0]       w_in_left,       // weight arriving from the right (transpose mode)
    output logic [DATA_W-1:0]       w_out_up,        // weight forwarded upward    (normal mode)
    output logic [DATA_W-1:0]       w_out_right,     // weight forwarded rightward (transpose mode)

    // Control
    input  logic                    load_w,          // 1 = latch weight, 0 = accumulate
    input  logic                    transpose_en     // 0 = weight enters from bottom, 1 = from right
);

// ── Internal registers ────────────────────────────────────────────
logic [DATA_W-1:0]    W_reg;     // stored weight (stationary for entire feed phase)
logic [DATA_W-1:0]    act_reg;   // registered activation (forwarded to right neighbor)
logic [DATA_W_OUT-1:0] psum_reg; // registered partial sum (forwarded downward)

// ── Combinational MAC ─────────────────────────────────────────────
// Multiply is done in full precision (2*DATA_W bits) to avoid
// intermediate overflow, then added to the incoming partial sum.
logic signed [2*DATA_W-1:0]   mac_mul;
logic signed [DATA_W_OUT-1:0] mac_res;

assign mac_mul = $signed(in_act) * $signed(W_reg);
assign mac_res = mac_mul + $signed(in_psum);
// ── Weight register ───────────────────────────────────────────────
// Latches on the first load_w=1 cycle and holds the value for the
// entire FEED_A phase. Source selected by transpose_en:
//   0 → w_in_down  (weights loaded row by row from the bottom boundary)
//   1 → w_in_left  (weights loaded column by column from the right boundary)
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        W_reg <= '0;
    else if (load_w && !transpose_en)
        W_reg <= w_in_down;
    else if (load_w && transpose_en)
        W_reg <= w_in_left;
end

// ── Activation and partial sum registers ─────────────────────────
// Active only during FEED_A (load_w=0).
// act_reg: pipelines the activation one cycle to the right neighbor.
// psum_reg: accumulates the MAC result and pipelines it downward.
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        act_reg  <= '0;
        psum_reg <= '0;
    end
    else if (!load_w) begin
        act_reg  <= in_act;
        psum_reg <= mac_res;
    end
end

// ── Output assignments ────────────────────────────────────────────
assign out_act     = act_reg;
assign out_psum    = psum_reg;

// Weight forwarding: only one direction drives a non-zero value.
// The inactive direction is tied to zero so it does not pollute
// neighbouring signals during transpose or normal mode.
assign w_out_up    = (transpose_en == 0) ? W_reg : '0;
assign w_out_right = (transpose_en)      ? W_reg : '0;

endmodule