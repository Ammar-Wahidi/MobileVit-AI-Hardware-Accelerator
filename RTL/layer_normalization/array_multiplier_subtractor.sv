module array_multiplier_subtractor #(
    parameter DATA_WIDTH = 32,
    parameter EMBED_DIM  = 32
) (
    input  wire signed [DATA_WIDTH-1:0]                            activation_in                        [0:EMBED_DIM-1]      ,
    input  wire signed [DATA_WIDTH-1:0]                            mean                                                      ,
    input  wire                                                    array_multiplier_subtractor_in_en                         ,
    output wire        [(2*DATA_WIDTH)-1:0]                        mul_out                              [0:EMBED_DIM-1]      ,   
    output wire signed [(DATA_WIDTH)-1:0]                          sub_out                              [0:EMBED_DIM-1]      ,   
    output wire                                                    array_multiplier_subtractor_out_en     
);
    
    assign array_multiplier_subtractor_out_en = array_multiplier_subtractor_in_en  ;
    genvar  i                                                                      ;
    generate
     for (i = 0  ; i<EMBED_DIM ; i = i+1 ) begin : gen_calc_loop
         assign sub_out[i] = (activation_in[i] - mean)                             ;
         assign mul_out[i] = sub_out[i] * sub_out[i]                               ;
     end
    endgenerate

   //always @(*) begin
   // for (i = 0 ; i < EMBED_DIM ; i= i+1) begin
   //     sub_out[i] = (activation_in[i] - mean)                             ;
   //     mul_out[i] = (activation_in[i] - mean) * (activation_in[i] - mean) ;
   // end
   //end
endmodule