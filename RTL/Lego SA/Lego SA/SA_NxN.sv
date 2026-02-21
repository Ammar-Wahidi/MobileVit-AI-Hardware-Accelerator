module SA_NxN #(
    parameter DATA_W = 8, parameter DATA_W_OUT = 32, parameter N_SIZE = 16
)(
input  logic                    clk                 ,
input  logic                    rst_n               ,
input  logic [DATA_W-1:0]       act_in  [N_SIZE]        ,     
input  logic [DATA_W-1:0]       weight_in [N_SIZE]      ,
input  logic                    load_w              ,    
input  logic                    transpose_en        ,    
  
output logic [DATA_W_OUT-1:0]   psum_out[N_SIZE]        
);

// Internal interconnect signals
logic   [DATA_W-1:0]        act_sig             [N_SIZE][N_SIZE+1]          ;       // extra column for right output
logic   [DATA_W-1:0]        weight_D_sig        [N_SIZE+1][N_SIZE]          ;
logic   [DATA_W-1:0]        weight_L_sig        [N_SIZE][N_SIZE+1]          ;
logic   [DATA_W_OUT-1:0]    psum_sig            [N_SIZE+1][N_SIZE]          ;       // extra row for bottom output




genvar k ;
genvar i,j ;

generate;
    for (k=0 ; k<N_SIZE; k++)
    begin
        assign act_sig[k][0]    = act_in[k] ;
        assign psum_sig[0][k]   = '0    ;
        assign weight_D_sig[N_SIZE][k] = weight_in[k] ;
        assign weight_L_sig[k][N_SIZE] = weight_in[k]; 
    end
endgenerate

generate;
    for (i = 0 ;i < N_SIZE;i++) begin :Row 
        for (j = 0 ;j < N_SIZE; j++) begin :COl
            PE #(.DATA_W(DATA_W),.DATA_W_OUT(DATA_W_OUT)) u_pe (
                .clk(clk),
                .rst_n(rst_n),
                .in_act(act_sig[i][j]),
                .in_psum(psum_sig[i][j]),
                .w_in_down(weight_D_sig[i+1][j]),
                .w_in_left(weight_L_sig[i][j+1]),
                .load_w(load_w),
                .transpose_en(transpose_en),
                .out_act(act_sig[i][j+1]),
                .out_psum(psum_sig[i+1][j]),
                .w_out_up(weight_D_sig[i][j]),
                .w_out_right(weight_L_sig[i][j])
            );
        end
    end

    for (j=0; j<N_SIZE; j++) assign psum_out[j] = psum_sig[N_SIZE][j];
endgenerate
endmodule