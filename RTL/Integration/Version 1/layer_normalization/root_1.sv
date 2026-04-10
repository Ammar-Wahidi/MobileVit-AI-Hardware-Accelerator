module root_1 #(
    parameter DATA_WIDTH = 32 ,
    parameter K_WIDTH    = 6  ,
    parameter M_WIDTH    = 17 ,
    parameter EMBED_DIM  = 32
) (
    input  wire                                              clk                                ,
    input  wire                                              rst_n                              ,
    input  wire                                              std_in_en                          ,
    input  wire         [(2*DATA_WIDTH)-1:0]                 vari                               ,
    output reg          [K_WIDTH-1       :0]                 k                                  ,
    output reg          [M_WIDTH-1       :0]                 m                                  ,
    output reg                                               root_1_out_valid            
);

wire [K_WIDTH-1:0] shifter     ;
reg  [K_WIDTH-1:0] counter     ;
reg  [K_WIDTH-1:0] counter_comb;
reg                            counter_en     ; 
reg                            counter_target ;
reg  [K_WIDTH-1       :0]      k_comb         ;
reg  [M_WIDTH-1       :0]      m_comb         ;
reg                            count          ;


assign shifter            = (counter - (DATA_WIDTH/2))    ;

always @(posedge clk , negedge rst_n) begin
    if (~rst_n) begin
        k <= 'd0 ;
        m <= 'd0 ;
    end else begin
        k <= k_comb ;
        m <= m_comb ;
    end
end

always @(posedge clk , negedge rst_n) begin
    if (~rst_n) begin
        root_1_out_valid <= 1'b0 ;
    end else if (counter_target && (!count)) begin
        root_1_out_valid <= 1'b1 ;
    end else begin
        root_1_out_valid <= 1'b0 ;
    end
end

always @(posedge clk , negedge rst_n) begin
    if (~rst_n) begin
        counter <= (2*DATA_WIDTH - 1) ;
    end else  begin
        counter <= counter_comb   ;
    end 
end


always @(posedge clk , negedge rst_n) begin
    if (~rst_n) begin
        count <= 1'b0           ;
    end else  begin
        count <= counter_target ;
    end
end


always @(*) begin
    if (std_in_en) begin
        counter_comb = counter - 1'b1 ;
    end else if (counter_en && (!counter_target)) begin
        counter_comb = counter - 1'b1 ;
    end else if (!counter_en) begin
        counter_comb = (2*DATA_WIDTH - 1)        ;
    end else begin
        counter_comb = counter     ;
    end
end

always @(*) begin
   if (vari[counter]) begin
       counter_target = 1'b1 ;
   end else begin
       counter_target = 1'b0 ;
   end
end

always @(*) begin
    if (counter_target) begin
        m_comb = vari >> shifter ;
        k_comb = shifter         ;
    end else begin
        m_comb = m               ;
        k_comb = k               ;
    end
end

always @(*) begin
    if (std_in_en) begin
        counter_en = 1'b1 ;
    end else if ((counter != (2*DATA_WIDTH -1)) && (!counter_target)) begin
        counter_en = 1'b1 ;
    end else begin
        counter_en = 1'b0 ;
    end
end


endmodule