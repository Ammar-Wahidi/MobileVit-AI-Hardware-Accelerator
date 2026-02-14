module Lego_SA_tb ();
parameter DATA_W = 8 ; parameter DATA_W_OUT = 32 ;

logic clk ;
logic rst_n ;
logic valid_in ;
logic [DATA_W-1:0]       act_in  [64] ;
logic [DATA_W-1:0]       weight_in  [64] ;
logic [1:0]              TYPE_Lego ;
logic                    load_w ;
logic                    transpose_en ;
logic [7:0] y_input_size;
logic [DATA_W_OUT-1:0]   psum_out[64] ;
logic                    load_w_finished   ;  
logic                    valid_out ;

Lego_Systolic_Array #(.DATA_W(DATA_W),.DATA_W_OUT(DATA_W_OUT)) Lego_SA (
.clk(clk),
.rst_n(rst_n),
.valid_in(valid_in),
.act_in(act_in),
.weight_in(weight_in),
.TYPE_Lego(TYPE_Lego),
.load_w(load_w),
.transpose_en(transpose_en),
.y_input_size(y_input_size),
.psum_out(psum_out),
.load_w_finished(load_w_finished),
.valid_out(valid_out)
);

initial clk = 0 ;
always #10 clk = ~clk ;

initial 
begin
    rst_n = 0 ;
    valid_in = 0 ;
    act_in = '{default:'0} ;
    weight_in = '{default:'0} ;
    TYPE_Lego = 0 ;
    load_w = 0 ;
    transpose_en = 0;
    y_input_size = 16;
    #15; // negedge
    rst_n = 1 ;
    valid_in = 1 ;
    TYPE_Lego = 1 ;
    transpose_en = 0;
    load_w = 1 ;
    weight_in = '{default:'b1} ;
    wait(load_w_finished);
    #5;
    load_w = 0 ;
    act_in = '{default:'b1} ;
    wait(valid_out);
    $display(psum_out);
    #200;
    $stop;
    


end

endmodule