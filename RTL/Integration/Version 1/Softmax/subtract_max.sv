module subtract_max #(
    parameter DATA_WIDTH = 32,
    parameter VECTOR_SIZE = 8
)
(
    input  logic signed [DATA_WIDTH-1:0] vec [VECTOR_SIZE],
    input  logic signed [DATA_WIDTH-1:0] max_val,
    output logic signed [DATA_WIDTH-1:0] shifted [VECTOR_SIZE]
);

integer i;

always_comb begin
    for (i=0;i<VECTOR_SIZE;i++)
        shifted[i] = vec[i] - max_val;
end

endmodule