module exp_sum
#(
    parameter DATA_WIDTH = 32,
    parameter VECTOR_SIZE = 8
)
(
    input  logic signed [DATA_WIDTH-1:0] exp_vec [VECTOR_SIZE],
    output logic signed [DATA_WIDTH-1:0] sum
);

integer i;

always_comb begin
    sum = 0;
    for (i=0; i<VECTOR_SIZE; i++)
        sum += exp_vec[i];
end

endmodule