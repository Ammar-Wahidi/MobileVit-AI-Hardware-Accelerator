module VARI #(
    parameter DATA_WIDTH = 32,
    parameter EMBED_DIM  = 32
) (
    input          [(2*DATA_WIDTH)+($clog2(EMBED_DIM))-1:0]  sum_2_in                          ,
    input                                                    vari_in_en                        ,
    output         [(2*DATA_WIDTH)-1:0]                      vari                              ,      
    output                                                   vari_out_en                             
);
  assign vari        = sum_2_in >>> $clog2(EMBED_DIM) ;
  assign vari_out_en = vari_in_en                     ;
endmodule