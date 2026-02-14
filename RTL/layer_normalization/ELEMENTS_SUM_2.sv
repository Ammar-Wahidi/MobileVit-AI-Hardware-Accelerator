module ELEMENTS_SUM_2 #(
    parameter DATA_WIDTH = 32,
    parameter EMBED_DIM  = 32
) (
    input  wire                                                    clk                                ,
    input  wire                                                    rst_n                              ,
    input  wire                                                    sum_en_2                           ,
    input  wire signed [(2*DATA_WIDTH)-1:0]                        mul_in         [0:EMBED_DIM-1]     ,
    output reg  signed [(2*DATA_WIDTH)+($clog2(EMBED_DIM))-1:0]    sum_2_out                          ,
    output reg                                                     sum_2_out_valid                 
);
   reg  signed [(2*DATA_WIDTH):0]                        sum_layer_1          [0:(EMBED_DIM/2)-1 ]  ;
   reg  signed [(2*DATA_WIDTH)+1:0]                      sum_layer_2          [0:(EMBED_DIM/4)-1 ]  ;
   reg  signed [(2*DATA_WIDTH)+2:0]                      sum_layer_3          [0:(EMBED_DIM/8)-1 ]  ;
   reg  signed [(2*DATA_WIDTH)+3:0]                      sum_layer_4          [0:(EMBED_DIM/16)-1]  ;
   reg  signed [(2*DATA_WIDTH):0]                        sum_layer_1_comb     [0:(EMBED_DIM/2)-1 ]  ;
   reg  signed [(2*DATA_WIDTH)+1:0]                      sum_layer_2_comb     [0:(EMBED_DIM/4)-1 ]  ;
   reg  signed [(2*DATA_WIDTH)+2:0]                      sum_layer_3_comb     [0:(EMBED_DIM/8)-1 ]  ;
   reg  signed [(2*DATA_WIDTH)+3:0]                      sum_layer_4_comb     [0:(EMBED_DIM/16)-1]  ;
   reg  signed [(2*DATA_WIDTH)+($clog2(EMBED_DIM))-1:0]  sum_2_out_comb                             ;
   reg         [3:0]                                     counter                                    ;
   wire                                                  counter_en                                 ;
   integer                                               i , j , k , q , w , e                      ;

   assign counter_en = (sum_en_2) ? 1'b1 : (counter != 'd6) ? 1'b1 : 1'b0 ;

// output sum register 

always @(posedge clk , negedge rst_n) begin
    if (~rst_n) begin
        sum_2_out <= 'd0            ;
    end else begin
        sum_2_out <= sum_2_out_comb ; 
    end
end


//sum layers registers as adder tree working
  always @(posedge clk , negedge rst_n) begin
    if (~rst_n) begin
        for (i = 0 ; i < (EMBED_DIM/2) ; i = i+1  ) begin
            sum_layer_1[i] <= 'd0 ;
        end
        for (i = 0 ; i < (EMBED_DIM/4) ; i = i+1  ) begin
            sum_layer_2[i] <= 'd0 ;
        end
        for (i = 0 ; i < (EMBED_DIM/8) ; i = i+1  ) begin
            sum_layer_3[i] <= 'd0 ;
        end
        for (i = 0 ; i < (EMBED_DIM/16) ; i = i+1  ) begin
            sum_layer_4[i] <= 'd0 ;
        end
    end else begin
        sum_layer_1  <= sum_layer_1_comb ;  
        sum_layer_2  <= sum_layer_2_comb ;  
        sum_layer_3  <= sum_layer_3_comb ;  
        sum_layer_4  <= sum_layer_4_comb ;  
    end
  end


 // counter for 5 clock stages for adder tree as it is 32 elements 
  always @(posedge clk , negedge rst_n) begin
    if (~rst_n) begin
        counter <= 'd6 ;
    end else if(sum_en_2) begin
        counter <= 0 ;
    end else if (counter_en) begin
        counter <= counter + 1'b1 ;
    end else begin
        counter <= 'd6 ;
    end
  end

 // output valid bit 
  always @(posedge clk , negedge rst_n) begin
    if (~rst_n) begin
        sum_2_out_valid <= 1'b0  ; 
    end else if (counter == 'd5) begin
        sum_2_out_valid <= 1'b1  ;
    end else begin
        sum_2_out_valid <= 1'b0  ;
    end
  end

// layer 1 for adder tree
 always @(*) begin
    k = 0 ;
    if (sum_en_2) begin
        for (j = 0  ; j < (EMBED_DIM) ; j = j  + 2 ) begin
           sum_layer_1_comb[k] = mul_in[j] + mul_in[j+1] ;
           k                   = k + 1                                 ; 
        end 
    end else begin
           sum_layer_1_comb    = sum_layer_1                           ; 
           k = 0 ;
        end
 end

// layer 2 for adder tree
 always @(*) begin
    q = 0 ;
   for (j = 0  ; j < (EMBED_DIM/2) ; j = j  + 2 ) begin
           sum_layer_2_comb[q] = sum_layer_1[j] + sum_layer_1[j+1] ;
           q                   = q + 1                             ; 
        end
 end


// layer 3 for adder tree
  always @(*) begin
    w = 0 ;
   for (j = 0  ; j < (EMBED_DIM/4) ; j = j  + 2 ) begin
           sum_layer_3_comb[w] = sum_layer_2[j] + sum_layer_2[j+1] ;
           w                   = w + 1                             ; 
        end
 end


// layer 4 for adder tree
  always @(*) begin
    e = 0 ;
   for (j = 0  ; j < (EMBED_DIM/8) ; j = j  + 2 ) begin
           sum_layer_4_comb[e] = sum_layer_3[j] + sum_layer_3[j+1] ;
           e                   = e + 1                             ; 
        end
 end

always @(*) begin
    if (counter == 4'd4) begin
        sum_2_out_comb  = sum_layer_4[0] + sum_layer_4[1] ;
    end else begin
        sum_2_out_comb  = sum_2_out                       ;
    end
end
endmodule