// ================================================================
//  SA_NxN — N x N Weight-Stationary Systolic Array
//
//  A 2-D mesh of N_SIZE x N_SIZE PE instances. Each PE holds one
//  weight value and computes one MAC per cycle during FEED_A.
//
//  Data flow:
//    Activations  : enter from the left column, shift right each cycle.
//    Partial sums : enter as zero from the top row, shift downward,
//                   accumulate through each column of PEs, exit at the
//                   bottom row as psum_out[].
//    Weights      : enter from the bottom boundary (normal) or right
//                   boundary (transpose) during LOAD_W, then stay
//                   fixed inside each PE for the entire FEED_A phase.
//
//  Weight loading (controlled by load_w and transpose_en):
//    Normal    (transpose_en=0): weight_in[j] feeds the bottom boundary
//              of column j and propagates upward one row per cycle.
//              After N_SIZE cycles, PE[row][col] holds W[row][col].
//    Transpose (transpose_en=1): weight_in[i] feeds the right boundary
//              of row i and propagates leftward one column per cycle.
//              After N_SIZE cycles, PE[row][col] also holds W[row][col]
//              because the caller drives column k of the weight matrix
//              on tick k, matching the column-wise entry.
//
//  Interconnect signals:
//    act_sig    [row][col] : activation at the input of PE[row][col-1]
//                            (col 0 = act_in, col N = discarded right output)
//    psum_sig   [row][col] : partial sum at the input of PE[row-1][col]
//                            (row 0 = zero, row N = psum_out)
//    weight_D_sig[row][col]: weight propagating upward between rows
//                            (row N = weight_in boundary, row 0 = discarded top output)
//    weight_L_sig[row][col]: weight propagating rightward between columns
//                            (col N = weight_in boundary, col 0 = discarded left output)
//
//  Parameters:
//    DATA_W     : activation / weight bit-width  (default 8)
//    DATA_W_OUT : partial sum bit-width          (default 32)
//    N_SIZE     : array dimension                (default 16)
//
// ================================================================

module SA_NxN #(
    parameter DATA_W     = 8,
    parameter DATA_W_OUT = 32,
    parameter N_SIZE     = 16
)(
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic [DATA_W-1:0]       act_in    [N_SIZE],   // activation row fed into the left edge
    input  logic [DATA_W-1:0]       weight_in [N_SIZE],   // weight bus: column (normal) or row (transpose)
    input  logic                    load_w,               // 1 = weight-load phase, 0 = accumulate phase
    input  logic                    transpose_en,         // 0 = weights enter from bottom, 1 = from right

    output logic [DATA_W_OUT-1:0]   psum_out  [N_SIZE]    // accumulated results from the bottom row
);

// ── Internal interconnect ─────────────────────────────────────────
// Extra column (+1) on act_sig  holds the unused right-exit outputs.
// Extra row    (+1) on psum_sig holds the zero injection at the top.
// Extra row    (+1) on weight_D_sig holds the bottom boundary injection.
// Extra column (+1) on weight_L_sig holds the right boundary injection.
logic [DATA_W-1:0]    act_sig      [N_SIZE][N_SIZE+1];
logic [DATA_W-1:0]    weight_D_sig [N_SIZE+1][N_SIZE];
logic [DATA_W-1:0]    weight_L_sig [N_SIZE][N_SIZE+1];
logic [DATA_W_OUT-1:0] psum_sig    [N_SIZE+1][N_SIZE];

genvar k;
genvar i, j;

// ── Boundary conditions ───────────────────────────────────────────
// Connect the external input buses to the array boundary nodes.
// act_in[k]    feeds the leftmost cell of row k.
// psum_sig[0]  is zero so the first PE row accumulates from scratch.
// weight_D_sig[N_SIZE][k] is the bottom boundary for normal loading.
// weight_L_sig[k][N_SIZE] is the right boundary for transpose loading.
generate
    for (k = 0; k < N_SIZE; k++) begin
        assign act_sig[k][0]           = act_in[k];
        assign psum_sig[0][k]          = '0;
        assign weight_D_sig[N_SIZE][k] = weight_in[k];
        assign weight_L_sig[k][N_SIZE] = weight_in[k];
    end
endgenerate

// ── PE array instantiation ────────────────────────────────────────
// PE[i][j] sits at row i, column j.
// Activations flow left to right:  act_sig[i][j] → PE → act_sig[i][j+1]
// Partial sums flow top to bottom: psum_sig[i][j] → PE → psum_sig[i+1][j]
// Weights (normal)    flow bottom to top: weight_D_sig[i+1][j] → PE → weight_D_sig[i][j]
// Weights (transpose) flow right to left: weight_L_sig[i][j+1] → PE → weight_L_sig[i][j]
generate
    for (i = 0; i < N_SIZE; i++) begin : Row
        for (j = 0; j < N_SIZE; j++) begin : COl
            PE #(
                .DATA_W    (DATA_W    ),
                .DATA_W_OUT(DATA_W_OUT)
            ) u_pe (
                .clk         (clk                ),
                .rst_n       (rst_n              ),
                .in_act      (act_sig[i][j]      ),
                .in_psum     (psum_sig[i][j]     ),
                .w_in_down   (weight_D_sig[i+1][j]),
                .w_in_left   (weight_L_sig[i][j+1]),
                .load_w      (load_w             ),
                .transpose_en(transpose_en       ),
                .out_act     (act_sig[i][j+1]   ),
                .out_psum    (psum_sig[i+1][j]  ),
                .w_out_up    (weight_D_sig[i][j] ),
                .w_out_right (weight_L_sig[i][j] )
            );
        end
    end

    // Collect the bottom-row partial sums as the array output.
    for (j = 0; j < N_SIZE; j++)
        assign psum_out[j] = psum_sig[N_SIZE][j];
endgenerate

endmodule