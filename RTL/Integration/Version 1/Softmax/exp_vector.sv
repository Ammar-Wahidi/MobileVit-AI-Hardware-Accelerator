module exp_vector
#(
    parameter DATA_WIDTH = 32,
    parameter FRAC_BITS  = 16,
    parameter VECTOR_SIZE = 8
)
(
    input  logic signed [DATA_WIDTH-1:0] vec [VECTOR_SIZE],
    output logic signed [DATA_WIDTH-1:0] exp_vec [VECTOR_SIZE]
);

genvar i;

generate for (i=0;i<VECTOR_SIZE;i++) begin : EXP_BLOCK
        exp #(
            .DATA_WIDTH(DATA_WIDTH),
            .FRAC_BITS(FRAC_BITS)
        ) exp_inst (
            .x(vec[i]),
            .y(exp_vec[i])
        );
    end
endgenerate

endmodule