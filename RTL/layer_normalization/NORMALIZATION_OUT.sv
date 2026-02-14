module NORMALIZATION_OUT #(
    parameter DATA_WIDTH = 32,
    parameter K_WIDTH    = 5 ,
    parameter M_WIDTH    = 9 ,
    parameter EMBED_DIM  = 32
) (
    input  wire                                              clk                                         ,
    input  wire                                              rst_n                                       ,
    input  wire                                              norm_final_in_en                             ,
    input  wire signed [(DATA_WIDTH)-1:0]                    sub_in                [0:EMBED_DIM-1]       ,
    input  wire        [(DATA_WIDTH)-1:0]                    std_dev_inv                                 , //Q4.28 for odd k , Q18.14 for k is even
    input  wire        [K_WIDTH-1       :0]                  k                                           ,
    output reg         [(DATA_WIDTH)-1:0]                    normalized_output      [0:EMBED_DIM-1]      , 
    output reg                                               norm_final_out_valid                         
);
wire   [(2*DATA_WIDTH)-1:0]   normalized_output_comb   [0:EMBED_DIM-1]  ; ////Q28.36 for odd k , Q22.42 for k is even
integer  i ;
always @(posedge clk , negedge rst_n) begin
    if (~rst_n) begin
        norm_final_out_valid <= 1'b0 ;
        for (i = 0 ; i<EMBED_DIM ; i= i+1 ) begin
            normalized_output[i] <= 'd0 ;
        end
    end else if (norm_final_in_en && k[0]) begin
        norm_final_out_valid <= 1'b1 ;
        for (i = 0 ; i<EMBED_DIM ; i= i+1 ) begin
            normalized_output[i] <= {2'd0,normalized_output_comb[i][59:28]} ;
        end
    end else if (norm_final_in_en && ~k[0]) begin
        norm_final_out_valid <= 1'b1 ;
        for (i = 0 ; i<EMBED_DIM ; i= i+1 ) begin
            normalized_output[i] <= {2'd0,normalized_output_comb[i][63:34]} ;
        end
    end else begin
        norm_final_out_valid <= 1'b0 ;
    end
end

genvar j;

    generate
        for (j = 0; j < EMBED_DIM; j = j + 1) begin
            assign normalized_output_comb[j] = (std_dev_inv) * sub_in[j] ; 
        end
    endgenerate


endmodule