module MEAN #(
    parameter DATA_WIDTH = 32,
    parameter EMBED_DIM  = 32
) (

    input   signed [DATA_WIDTH+($clog2(EMBED_DIM))-1:0]  sum_1_in                          ,
    input                                                mean_in_en                        ,
    output  signed [DATA_WIDTH-1:0]                      mean                              ,      
    output                                               mean_out_en                             
);
  
  assign mean        = sum_1_in >>> $clog2(EMBED_DIM) ;
  assign mean_out_en = mean_in_en                     ;

endmodule