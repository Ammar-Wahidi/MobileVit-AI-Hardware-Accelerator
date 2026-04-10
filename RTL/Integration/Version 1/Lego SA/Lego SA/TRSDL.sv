// ================================================================
//  TRSDL — Triangular Register Shift Down Logic
//
//  Removes the diagonal skew from the partial sum outputs of an
//  NxN systolic array so that all N results for one activation
//  row appear on the output bus simultaneously.
//
//  Why skew exists on the output:
//    The systolic array is fed through TRSRL, which delays act[k]
//    by k cycles. As a result, the final accumulated partial sum
//    for column j finishes one cycle later than column j-1.
//    psum[j] is therefore ready j cycles after psum[0].
//
//  What TRSDL does:
//    It is structurally identical to TRSRL but operates on partial
//    sums instead of activations. Lane j delays psum_in[j] by j
//    cycles, so all N outputs align in time.
//
//  Before TRSDL, the caller reverses the column order (mirror),
//  feeds the reversed array into TRSDL, then reverses again on
//  the way out. This is how the top-level module compensates for
//  the decreasing skew (column N-1 needs the most delay, which
//  maps to lane 0 of TRSDL that has the most registers).
//
//  Implementation — triangular shift register chain:
//    Lane 0 : direct wire   (0 registers)
//    Lane 1 : 1 register    (delay = 1 cycle)
//    Lane k : k registers   (delay = k cycles)
//
//  Total register count = N*(N-1)/2  (same formula as TRSRL)
//
//  Parameters:
//    DATAWIDTH : data bit-width  (default 32, matches accumulator width)
//    N_SIZE    : number of lanes (default 16)
//
// ================================================================

module TRSDL #(
    parameter DATAWIDTH = 32,
    parameter N_SIZE    = 16
)(
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic [DATAWIDTH-1:0]    psum_in  [N_SIZE],  // skewed partial sum inputs
    output logic [DATAWIDTH-1:0]    psum_out [N_SIZE]   // de-skewed partial sum outputs
);

// Total number of shift registers needed across all lanes.
localparam NUM_OF_REGS = ((N_SIZE - 1) * N_SIZE) / 2;

// Flat register array for the full triangular shift chain (1-based indexing).
logic [DATAWIDTH-1:0] reg_shifted [1:NUM_OF_REGS];

// Intermediate de-skewed signals — psum[k] is psum_in[k] delayed by k cycles.
logic [DATAWIDTH-1:0] psum [N_SIZE];

// Lane 0 passes through with no delay.
// Lane 1 passes through one register.
assign psum[0] = psum_in[0];
assign psum[1] = reg_shifted[1];

genvar k, i_deptha;
genvar l;

// Build the triangular shift chain for lanes 1 through N_SIZE-1.
generate
    for (k = 1; k < N_SIZE; k++) begin

        // Base index for lane k in the flat register array.
        localparam int base = (k * (k - 1)) / 2;

        // First register in lane k: latches psum_in[k].
        always_ff @(posedge clk or negedge rst_n) begin : First_col_Resgs
            if (!rst_n)
                reg_shifted[(base) + 1] <= 0;
            else
                reg_shifted[(base) + 1] <= psum_in[k];
        end

        // Additional shift stages for lanes deeper than one register.
        if (k > 1) begin : DEPTH_LEVEL
            for (i_deptha = (base) + 2; i_deptha < ((base) + 1) + k; i_deptha++) begin
                always_ff @(posedge clk or negedge rst_n) begin
                    if (~rst_n)
                        reg_shifted[i_deptha] <= 0;
                    else
                        reg_shifted[i_deptha] <= reg_shifted[i_deptha - 1];
                end
            end
        end

        // Output of lane k is taken from the end of its shift chain.
        if (k > 1) begin
            assign psum[k] = reg_shifted[(base) + 1 + k - 1];
        end

    end
endgenerate

// Connect internal de-skewed signals to the output ports.
generate
    for (l = 0; l < N_SIZE; l++)
        assign psum_out[l] = psum[l];
endgenerate

endmodule