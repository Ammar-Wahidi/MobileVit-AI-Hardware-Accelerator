module SA_Batch #(
    parameter DATA_W     = 8,
    parameter DATA_W_OUT = 32,
    parameter Y_INPUT_SIZE = 8,
    parameter N_TILE     = 16
)(
    input   logic                       clk,
    input   logic                       rst_n,

    // Control Lego SA 
    input   logic                       valid_in_SA,
    input   logic [1:0]                 lego_type,
    input   logic [Y_INPUT_SIZE-1:0]    y_input_size,
    input   logic                       transpose_en,

    // Data inputs 
    input  logic [DATA_W-1:0]           act_in    [4*N_TILE],
    input  logic [DATA_W-1:0]           weight_in [4*N_TILE],

    // A,B Prams input 
    input  logic [Data_Width-1:0]   A [0:N-1], 
    input  logic [Data_Width-1:0]   B [0:N-1], 

    // Output 
    output logic                        valid_out,
    output logic [DATA_W_OUT-1:0]       out_row_SA_BA [4*N_TILE]

);

localparam TOTAL_W = 4 * N_TILE;   // full bus width
localparam FRAC_BITS = 8 ;
localparam N_PER_ROW = 64 ;

logic [DATA_W_OUT-1:0]      psum_out_row  [TOTAL_W] ;
logic                       valid_out_SA ;

Lego_SA #(
    .DATA_W(DATA_W),
    .DATA_W_OUT(DATA_W_OUT),
    .Y_INPUT_SIZE(Y_INPUT_SIZE),
    .N_TILE(N_TILE)
) u_SA (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(valid_in_SA),
    .lego_type(lego_type),
    .y_input_size(y_input_size),
    .transpose_en(transpose_en),
    .act_in(act_in),
    .weight_in(weight_in),
    .psum_out(psum_out_row),
    .valid_out(valid_out_SA),
    .busy()
);

Batch_Norm #(
    .Data_Width(DATA_W_OUT),
    .FRAC_BITS(FRAC_BITS)
    .N(N_PER_ROW)
) Batch_Normalization (
    .CLk(clk),
    .RST(rst_n),
    .in_row(psum_out_row),
    .INBatch_Valid(valid_out_SA),
    .A(A),
    .B(B),
    .out_row(out_row_SA_BA),
    .OutBatch_Valid(valid_out)
);


endmodule