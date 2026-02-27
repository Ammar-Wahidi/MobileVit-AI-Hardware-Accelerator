// ================================================================
//  TRSRL — Triangular Register Shift Right Logic
//
//  Skews N parallel input lanes so that lane[k] arrives at its
//  systolic array row k cycles later than lane[0].
//
//  This is the classic activation-skew stage used before an NxN
//  weight-stationary systolic array. Without it, all activation
//  rows would enter the array on the same cycle and collide.
//  With it, row 0 enters immediately, row 1 enters one cycle
//  later, row 2 two cycles later, and so on — matching the
//  diagonal wavefront of the weight-stationary computation.
//
//  Implementation — triangular shift register chain:
//    Lane 0 : direct wire   (0 registers, no delay)
//    Lane 1 : 1 register    (delay = 1 cycle)
//    Lane 2 : 2 registers   (delay = 2 cycles)
//    Lane k : k registers   (delay = k cycles)
//
//  Total register count = 0+1+2+...+(N-1) = N*(N-1)/2
//  This is stored flat in reg_shifted[1 : NUM_OF_REGS].
//
//  Indexing:
//    For lane k, the base index into reg_shifted is k*(k-1)/2.
//    reg_shifted[base+1]   : first register in lane k (latches act_in[k])
//    reg_shifted[base+2..k]: shift chain for the remaining k-1 registers
//    act[k]                = reg_shifted[base+k] (the last register output)
//
//  Parameters:
//    DATAWIDTH : data bit-width  (default 8)
//    N_SIZE    : number of lanes (default 16)
//
// ================================================================

module TRSRL #(
    parameter DATAWIDTH = 8,
    parameter N_SIZE    = 16
)(
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic [DATAWIDTH-1:0]    act_in  [N_SIZE],   // parallel activation input lanes
    output logic [DATAWIDTH-1:0]    act_out [N_SIZE]    // skewed activation output lanes
);

// Total number of shift registers needed for all lanes combined.
// Lane k needs k registers, so the total is sum(1..N-1) = N*(N-1)/2.
localparam NUM_OF_REGS = ((N_SIZE - 1) * N_SIZE) / 2;

// Flat register array holding the entire triangular shift chain.
// Indexed from 1 to NUM_OF_REGS (1-based to match the index formula).
logic [DATAWIDTH-1:0] reg_shifted [1:NUM_OF_REGS];

// Intermediate skewed signals — act[k] is act_in[k] delayed by k cycles.
logic [DATAWIDTH-1:0] act [N_SIZE];

// Lane 0 requires no delay — direct wire.
// Lane 1 requires exactly one register, which is reg_shifted[1].
assign act[0] = act_in[0];
assign act[1] = reg_shifted[1];

genvar k, i_deptha;
genvar l;

// Build the triangular shift chain for lanes 1 through N_SIZE-1.
generate
    for (k = 1; k < N_SIZE; k++) begin

        // Base index in reg_shifted for lane k.
        // Lane k occupies positions [base+1 .. base+k].
        localparam int base = (k * (k - 1)) / 2;

        // First register in lane k: captures act_in[k] directly.
        always_ff @(posedge clk or negedge rst_n) begin : First_col_Resgs
            if (!rst_n)
                reg_shifted[(base) + 1] <= 0;
            else
                reg_shifted[(base) + 1] <= act_in[k];
        end

        // Lanes longer than one register need an additional shift chain.
        // Each stage just passes the value from the previous stage.
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

        // The output of lane k is taken from the last register in its chain.
        if (k > 1) begin
            assign act[k] = reg_shifted[(base) + 1 + k - 1];
        end

    end
endgenerate

// Connect internal skewed signals to output ports.
generate
    for (l = 0; l < N_SIZE; l++)
        assign act_out[l] = act[l];
endgenerate

endmodule