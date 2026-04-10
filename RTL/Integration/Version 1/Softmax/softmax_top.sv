module softmax_top
#(
    parameter DATA_WIDTH = 32,
    parameter FRAC_BITS  = 16,
    parameter VECTOR_SIZE = 8
)
(
    input  logic  [DATA_WIDTH-1:0] input_vector [VECTOR_SIZE],
    output logic signed [DATA_WIDTH-1:0] output_vector [VECTOR_SIZE]
);

logic signed [DATA_WIDTH-1:0] max_val;
logic signed [DATA_WIDTH-1:0] shifted [VECTOR_SIZE];
logic signed [DATA_WIDTH-1:0] exp_vec [VECTOR_SIZE];
logic signed [DATA_WIDTH-1:0] sum_exp;
logic signed [DATA_WIDTH-1:0] inv_sum;

vector_max #(
    .DATA_WIDTH(DATA_WIDTH),
    .VECTOR_SIZE(VECTOR_SIZE)
) max_block (
    .vec(input_vector),
    .max_value(max_val)
);

subtract_max #(
    .DATA_WIDTH(DATA_WIDTH),
    .VECTOR_SIZE(VECTOR_SIZE)
) subtract_block (
    .vec(input_vector),
    .max_val(max_val),
    .shifted(shifted)
);

exp_vector #(
    .DATA_WIDTH(DATA_WIDTH),
    .FRAC_BITS(FRAC_BITS),
    .VECTOR_SIZE(VECTOR_SIZE)
) exp_block (
    .vec(shifted),
    .exp_vec(exp_vec)
);

exp_sum #(
    .DATA_WIDTH(DATA_WIDTH),
    .VECTOR_SIZE(VECTOR_SIZE)
) sum_block (
    .exp_vec(exp_vec),
    .sum(sum_exp)
);

reciprocal_nr #(
    .DATA_WIDTH(DATA_WIDTH),
    .FRAC_BITS(FRAC_BITS)
) reciprocal_block (
    .x(sum_exp),
    .y(inv_sum)
);

normalize #(
    .DATA_WIDTH(DATA_WIDTH),
    .FRAC_BITS(FRAC_BITS),
    .VECTOR_SIZE(VECTOR_SIZE)
) norm_block (
    .exp_vec(exp_vec),
    .inv_sum(inv_sum),
    .softmax(output_vector)
);

endmodule