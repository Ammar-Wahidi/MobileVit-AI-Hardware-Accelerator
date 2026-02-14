module layer_normalization_top #(
    parameter DATA_WIDTH = 32,
    parameter K_WIDTH    = 5 ,
    parameter M_WIDTH    = 9 ,
    parameter EMBED_DIM  = 32
) (
    input  wire                                              clk                                     ,
    input  wire                                              rst_n                                   ,
    input  wire                                              sum_en_1                                ,
    input  wire signed [DATA_WIDTH-1:0]                      activation_in [0:EMBED_DIM-1]           ,
    output wire        [(DATA_WIDTH)-1:0]                    normalized_output      [0:EMBED_DIM-1]  , 
    output wire                                              norm_final_out_valid                         
);

wire signed [DATA_WIDTH+($clog2(EMBED_DIM))-1:0]      sum_1_out                                         ;
wire                                                  sum_1_out_valid                                   ;
wire signed [DATA_WIDTH-1:0]                          mean                                              ;
wire                                                  mean_out_en                                       ;   
wire signed [(2*DATA_WIDTH)-1:0]                      mul_out                            [0:EMBED_DIM-1];
wire signed [(DATA_WIDTH)-1:0]                        sub_out                            [0:EMBED_DIM-1]; 
wire                                                  array_multiplier_subtractor_out_en                ;
wire signed [(2*DATA_WIDTH)+($clog2(EMBED_DIM))-1:0]  sum_2_out                                         ;
wire                                                  sum_2_out_valid                                   ;
wire signed [(2*DATA_WIDTH)-1:0]                      vari                                              ;
wire                                                  vari_out_en                                       ;
wire         [(DATA_WIDTH)-1:0]                       std_dev_inv                                       ;
wire                                                  root_final_out_valid                              ;
wire         [K_WIDTH-1       :0]                     k                                                 ; 

  ELEMENTS_SUM #(
    .DATA_WIDTH (DATA_WIDTH),
    .EMBED_DIM  (EMBED_DIM )
  ) elements_sum_1 (
       .clk             (clk            ),
       .rst_n           (rst_n          ),
       .sum_en_1        (sum_en_1       ),
       .activation_in   (activation_in  ),
       .sum_1_out       (sum_1_out      ),
       .sum_1_out_valid (sum_1_out_valid) 
 );

  MEAN #(
    .DATA_WIDTH (DATA_WIDTH),
    .EMBED_DIM  (EMBED_DIM )
  ) mean_1 (
       .sum_1_in    (sum_1_out       ),   
       .mean_in_en  (sum_1_out_valid ),
       .mean        (mean            ),
       .mean_out_en (mean_out_en     )     
 );


 array_multiplier_subtractor #(
    .DATA_WIDTH (DATA_WIDTH),
    .EMBED_DIM  (EMBED_DIM )
 ) array_multiplier_subtractor_block (
       .activation_in                      (activation_in                     ),
       .mean                               (mean                              ),
       .array_multiplier_subtractor_in_en  (mean_out_en                       ),
       .mul_out                            (mul_out                           ),
       .sub_out                            (sub_out                           ),
       .array_multiplier_subtractor_out_en (array_multiplier_subtractor_out_en) 
 );

 ELEMENTS_SUM_2 #(
     .DATA_WIDTH (DATA_WIDTH),
     .EMBED_DIM  (EMBED_DIM )
   ) elements_sum_2 (
        .clk             (clk                               ),
        .rst_n           (rst_n                             ),
        .sum_en_2        (array_multiplier_subtractor_out_en),
        .mul_in          (mul_out                           ),
        .sum_2_out       (sum_2_out                         ),
        .sum_2_out_valid (sum_2_out_valid                   ) 
  );


  VARI #(
     .DATA_WIDTH (DATA_WIDTH),
     .EMBED_DIM  (EMBED_DIM )
  ) VARI_block (
        .sum_2_in    (sum_2_out        ) ,
        .vari_in_en  (sum_2_out_valid  ) ,
        .vari        (vari             ) ,
        .vari_out_en (vari_out_en      )  
  );

  standard_deviation_inv #(
    .DATA_WIDTH (DATA_WIDTH),
    .K_WIDTH    (K_WIDTH   ),
    .M_WIDTH    (M_WIDTH   ),
    .EMBED_DIM  (EMBED_DIM )
  ) standard_deviation_inv_block (
       .clk                  (clk                 ),
       .rst_n                (rst_n               ),
       .std_in_en            (vari_out_en         ),
       .vari                 (vari                ),
       .std_dev_inv          (std_dev_inv         ),
       .root_final_out_valid (root_final_out_valid),
       .k                    (k                   )
 );


 NORMALIZATION_OUT #(
    .DATA_WIDTH (DATA_WIDTH),
    .K_WIDTH    (K_WIDTH   ),
    .M_WIDTH    (M_WIDTH   ),
    .EMBED_DIM  (EMBED_DIM )
 ) normalization_out_block (
       .clk                  (clk                 ),
       .rst_n                (rst_n               ),
       .norm_final_in_en     (root_final_out_valid),
       .sub_in               (sub_out             ),
       .std_dev_inv          (std_dev_inv         ),
       .k                    (k                   ),
       .normalized_output    (normalized_output   ),
       .norm_final_out_valid (norm_final_out_valid)
 );





endmodule