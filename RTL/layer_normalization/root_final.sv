module root_final #(
    parameter DATA_WIDTH = 32,
    parameter K_WIDTH    = 6 ,
    parameter M_WIDTH    = 17,
    parameter EMBED_DIM  = 32
) (
    input  wire                                              clk                                ,
    input  wire                                              rst_n                              ,
    input  wire                                              root_final_in_en                   ,
    input  wire        [(DATA_WIDTH/2)-1: 0]                 y0_1                               , 
    input  wire        [K_WIDTH-1       :0]                  k                                  ,
    output reg         [(DATA_WIDTH)-1:0]                    std_dev_inv                        , //Q4.28 for odd k , Q18.14 for k  even
    output reg                                               root_final_out_valid                         
);

wire [(DATA_WIDTH/2)-1: 0] constant    ;
reg  [K_WIDTH-1       :0]  shifter     ;
reg  [(DATA_WIDTH)-1:0]    std_dev_inv_comb ;
assign constant   = 16'h2D41           ;

always @(posedge clk , negedge rst_n) begin
    if (~rst_n) begin
        std_dev_inv <= 'd0 ;
        root_final_out_valid <= 'd0 ;
    end else if (root_final_in_en && k[0]) begin
        std_dev_inv <= std_dev_inv_comb        ;
        root_final_out_valid <= 1'b1 ;
    end else if (root_final_in_en && (~k[0])) begin
        std_dev_inv <= std_dev_inv_comb        ;
        root_final_out_valid <= 1'b1 ;
    end else begin
        root_final_out_valid <= 1'b0 ;
    end
end

always @(*) begin
    if (k[0]) begin
        shifter = ((k - 1'b1) >> 1'b1); 
    end else begin
        shifter = k >> 1'b1          ;
    end
end

always @(*) begin
    if (k[0]) begin
        std_dev_inv_comb = ((y0_1 >> shifter) * constant) ;
    end else if (~k[0]) begin
        std_dev_inv_comb = (y0_1 >> shifter)              ;
    end else begin
        std_dev_inv_comb = std_dev_inv                    ;
    end
end


endmodule