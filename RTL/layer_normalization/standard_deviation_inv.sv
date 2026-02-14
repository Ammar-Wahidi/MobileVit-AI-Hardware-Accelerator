module standard_deviation_inv #(
    parameter DATA_WIDTH = 32,
    parameter K_WIDTH    = 6,
    parameter M_WIDTH    = 17,
    parameter EMBED_DIM  = 32
) (
    input  wire                                              clk                                ,
    input  wire                                              rst_n                              ,
    input  wire                                              std_in_en                          ,
    input  wire         [(2*DATA_WIDTH)-1:0]                 vari                               ,
    output wire         [(DATA_WIDTH)-1:0]                   std_dev_inv                        , //Q4.28 for odd k , Q18.14 for k is even
    output wire                                              root_final_out_valid               ,
    output wire         [K_WIDTH-1       :0]                 k                
);


wire [M_WIDTH-1       :0]                               m                ;
wire                                                    root_1_out_valid ;
wire                                                    root_2_out_valid ;
wire [(DATA_WIDTH/2)-1: 0]                              y0_1             ;


 root_1 #(
    .DATA_WIDTH (DATA_WIDTH),
    .K_WIDTH    (K_WIDTH   ),
    .M_WIDTH    (M_WIDTH   ),
    .EMBED_DIM  (EMBED_DIM )
 ) uut1 (
       .clk              (clk              ), 
       .rst_n            (rst_n            ),
       .vari             (vari             ), 
       .std_in_en        (std_in_en        ), 
       .k                (k                ), 
       .m                (m                ), 
       .root_1_out_valid (root_1_out_valid ) 
 );


  root_2 #(
    .DATA_WIDTH (DATA_WIDTH),
    .K_WIDTH    (K_WIDTH   ),
    .M_WIDTH    (M_WIDTH   ),
    .EMBED_DIM  (EMBED_DIM )
 ) uut2 (
        .clk              (clk             ),
        .rst_n            (rst_n           ),
        .root_2_in_en     (root_1_out_valid),
        .y0_1             (y0_1            ),
        .root_2_out_valid (root_2_out_valid),
        .k                (k               ),
        .m                (m               )
 );

  root_final #(
    .DATA_WIDTH (DATA_WIDTH),
    .K_WIDTH    (K_WIDTH   ),
    .M_WIDTH    (M_WIDTH   ),
    .EMBED_DIM  (EMBED_DIM )
 ) uut3 (
        .clk                  ( clk                    ),   
        .rst_n                ( rst_n                  ),   
        .root_final_in_en     ( root_2_out_valid       ),   
        .y0_1                 ( y0_1                   ),   
        .k                    ( k                      ),   
        .std_dev_inv          ( std_dev_inv            ),   
        .root_final_out_valid ( root_final_out_valid   )   
 );

endmodule