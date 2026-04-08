module SA_Swish #(
    parameter int DATA_W       = 8,
    parameter int DATA_W_OUT   = 32,
    parameter int N_TILE       = 16,  
    parameter int FRACT_BITS   = 8,
    parameter int Y_INPUT_SIZE = 8
) (
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  logic [1:0] lego_type,
    input  logic trans_en,
    
    input  logic [DATA_W-1:0]     act_bus    [4*N_TILE],
    input  logic [DATA_W-1:0]     weight_bus [4*N_TILE],
    output logic [DATA_W_OUT-1:0] d_out      [4*N_TILE],
    output logic valid_out
);

    logic busy;
    logic valid_out_lego;
    
    logic [DATA_W_OUT-1:0] result_bus [4*N_TILE]; 

    Lego_SA #(
        .DATA_W(DATA_W),
        .DATA_W_OUT(DATA_W_OUT),
        .N_TILE(N_TILE),
        .Y_INPUT_SIZE(Y_INPUT_SIZE)
    ) u_lego_sa (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .lego_type(lego_type),
        .y_input_size(8'd0), 
        .transpose_en(trans_en),
        .act_in(act_bus),
        .weight_in(weight_bus),
        .psum_out(result_bus), 
        .valid_out(valid_out_lego),
        .busy(busy)
    );

    swish_array #(
        .N(4*N_TILE),
        .WIDTH(DATA_W_OUT),
        .FRACT_BITS(FRACT_BITS)
    ) u_swish_array (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_out_lego), 
        .x(result_bus),             // Direct connect!
        .y(d_out),
        .valid_out(valid_out)
    );

endmodule