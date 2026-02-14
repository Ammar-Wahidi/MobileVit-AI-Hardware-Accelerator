
module Lego_control_unit #()(
input   logic           clk                 ,
input   logic           rst_n               ,
input   logic           valid_in            ,
input   logic [7:0]     y_input_size        ,

output  logic           load_w_finished     ,
output  logic           valid_out
);

localparam N_CYCLES_LOAD_W = 16 + 1 ;

logic [10:0]     count ;
logic count_finish;

always_ff @(posedge clk or negedge rst_n)
begin
    if (~rst_n)
    begin
        count <= 0 ;
    end
    else if(valid_in)
    begin
        count <= count + 1 ;
    end
    else 
    begin
        count <= (count == count_finish)? 0: count+1 ;
    end
end

assign load_w_finished = (count == N_CYCLES_LOAD_W);
assign valid_out =
    (count >= N_CYCLES_LOAD_W + y_input_size + 30) &&
    (count <  N_CYCLES_LOAD_W + 2*y_input_size + 30);
assign count_finish = (count > N_CYCLES_LOAD_W + 2*y_input_size + 30);



endmodule