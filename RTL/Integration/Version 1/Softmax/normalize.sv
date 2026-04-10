module normalize
#(
    parameter DATA_WIDTH = 32,
    parameter FRAC_BITS  = 16,
    parameter VECTOR_SIZE = 8
)
(
    input  logic signed [DATA_WIDTH-1:0] exp_vec [VECTOR_SIZE],
    input  logic signed [DATA_WIDTH-1:0] inv_sum,
    output logic signed [DATA_WIDTH-1:0] softmax [VECTOR_SIZE]
);

integer i;

always_comb begin
    for (i = 0; i < VECTOR_SIZE; i++)
        // Use 64-bit multiplication to avoid overflow
        softmax[i] = ( (64'(exp_vec[i]) * inv_sum) >>> FRAC_BITS );
end

endmodule