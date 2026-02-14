
module Lego_Systolic_Array #(
    parameter DATA_W = 8, parameter DATA_W_OUT = 32
)(
input  logic                    clk                 ,
input  logic                    rst_n               ,
input  logic                    valid_in            ,   // Valid in inputs 
input  logic [DATA_W-1:0]       act_in  [64]        ,   // left edge activations
input  logic [DATA_W-1:0]       weight_in  [64]     ,   // weights for each PE   
input  logic [1:0]              TYPE_Lego           ,   
input  logic                    load_w              ,   // load weight phase
input  logic                    transpose_en        ,   // Weight Transpose enable for PE
input  logic [7:0]              y_input_size        ,
output logic [DATA_W_OUT-1:0]   psum_out[64]        ,   // bottom edge partial sums
output logic                    load_w_finished     ,
output logic                    valid_out               // Valid out psum Col
);

logic   [DATA_W-1:0]        act_reg [64]        ;
logic   [DATA_W-1:0]        act_reg_RU [16]     ;
logic   [DATA_W-1:0]        act_reg_LU [16]     ;
logic   [DATA_W-1:0]        act_reg_RD [16]     ;
logic   [DATA_W-1:0]        act_reg_LD [16]     ;

logic   [DATA_W-1:0]        W_reg    [64]     ;
logic   [DATA_W-1:0]        W_reg_RU [16]     ;
logic   [DATA_W-1:0]        W_reg_LU [16]     ;
logic   [DATA_W-1:0]        W_reg_RD [16]     ;
logic   [DATA_W-1:0]        W_reg_LD [16]     ;

logic   [DATA_W_OUT-1:0]    psum[64]      ;
logic   [DATA_W_OUT-1:0]    psum_reg[64]      ;
logic   [DATA_W_OUT-1:0]    psum_reg_add_1[32]     ;
logic   [DATA_W_OUT-1:0]    psum_reg_add_2[16]  ;
logic   [DATA_W_OUT-1:0]    psum_RU [16]      ;
logic   [DATA_W_OUT-1:0]    psum_LU [16]      ;
logic   [DATA_W_OUT-1:0]    psum_RD [16]      ;
logic   [DATA_W_OUT-1:0]    psum_LD [16]      ;


assign psum[0:15] = psum_RU ;
assign psum[16:31]= psum_LU ;
assign psum[32:47]= psum_RD ;
assign psum[48:63]= psum_LD ;

assign psum_out = psum_reg ;


always_ff @(posedge clk or negedge rst_n)
begin
    if (~rst_n)
    begin
        act_reg <= '{default:'0}    ;
        W_reg   <= '{default:'0}    ;
    end
    else if (load_w & valid_in)
    begin
        W_reg   <= weight_in    ;
    end
    else if (~load_w & valid_in)
    begin
        act_reg <= act_in   ;
    end
    else 
    begin
        W_reg   <= '{default:'0}    ;
        act_reg <= '{default:'0}    ;
    end
end

always_comb
begin
    act_reg_RU = '{default:'0}  ;
    act_reg_LU = '{default:'0}  ;
    act_reg_RD = '{default:'0}  ;
    act_reg_LD = '{default:'0}  ;
    W_reg_RU = '{default:'0}    ;
    W_reg_LU = '{default:'0}    ;
    W_reg_RD = '{default:'0}    ;
    W_reg_LD = '{default:'0}    ;
    if (load_w & valid_in)
    begin
        W_reg_RU = W_reg[0:15]  ;
        W_reg_LU = W_reg[16:31]  ;
        W_reg_RD = W_reg[32:47]  ;
        W_reg_LD = W_reg[48:63]  ;
    end
    else if (~load_w & valid_in)
    begin
        case (TYPE_Lego)
            0:begin
                act_reg_RU = act_reg[0:15]  ;
                act_reg_LU = act_reg[0:15]  ;
                act_reg_RD = act_reg[0:15]  ;
                act_reg_LD = act_reg[0:15]  ;
            end
            1:begin
                act_reg_RU = act_reg[0:15]  ;
                act_reg_LU = act_reg[0:15]  ;
                act_reg_RD = act_reg[16:31]  ;
                act_reg_LD = act_reg[16:31]  ;
            end
            2:begin
                act_reg_RU = act_reg[0:15]  ;
                act_reg_LU = act_reg[32:47]  ;
                act_reg_RD = act_reg[16:31]  ;
                act_reg_LD = act_reg[48:63]  ;
            end
            default:begin
                act_reg_RU = '{default:'0}   ;
                act_reg_LU = '{default:'0}   ;
                act_reg_RD = '{default:'0}   ;
                act_reg_LD = '{default:'0}   ;
            end
        endcase
    end
    end

always_ff @(posedge clk or negedge rst_n)
begin
if (~rst_n)
begin
    psum_reg <= '{default:'0}    ;
end
else if (~load_w)
begin
    case(TYPE_Lego)
        0:begin
            psum_reg <= psum ;
        end
        1:begin
            psum_reg[0:31] <= psum_reg_add_1;
        end
        2:begin
            psum_reg[0:15] <= psum_reg_add_2;
        end
    endcase
end
else 
begin
    psum_reg <= '{default:'0}    ;
end
end

genvar p ;
generate;
for (p = 0 ; p <32 ; p++)
    assign psum_reg_add_1[p] = psum[p] + psum[p+32] ;
for (p = 0 ; p <16 ; p++)
    assign psum_reg_add_2[p] = psum[p] + psum[p+16] +  psum[p+32] + psum[p+48] ;
endgenerate

Lego_control_unit #() control_unit (
.clk(clk),
.rst_n(rst_n),
.valid_in(valid_in),
.y_input_size(y_input_size),
.load_w_finished(load_w_finished),
.valid_out(valid_out)
);

SA_16x16_top #(.DATA_W(DATA_W),.DATA_W_OUT(DATA_W_OUT)) SA0_RightUp_corner (
.clk(clk),
.rst_n(rst_n),
.act_in(act_reg_RU),
.weight_in(W_reg_RU),
.load_w(load_w),
.transpose_en(transpose_en),
.psum_out(psum_RU)
);

SA_16x16_top #(.DATA_W(DATA_W),.DATA_W_OUT(DATA_W_OUT)) SA1_LeftUp_corner (
.clk(clk),
.rst_n(rst_n),
.act_in(act_reg_LU),
.weight_in(W_reg_LU),
.load_w(load_w),
.transpose_en(transpose_en),
.psum_out(psum_LU)
);

SA_16x16_top #(.DATA_W(DATA_W),.DATA_W_OUT(DATA_W_OUT)) SA2_RightDown_corner (
.clk(clk),
.rst_n(rst_n),
.act_in(act_reg_RD),
.weight_in(W_reg_RD),
.load_w(load_w),
.transpose_en(transpose_en),
.psum_out(psum_RD)
);

SA_16x16_top #(.DATA_W(DATA_W),.DATA_W_OUT(DATA_W_OUT)) SA3_LeftDown_corner (
.clk(clk),
.rst_n(rst_n),
.act_in(act_reg_LD),
.weight_in(W_reg_LD),
.load_w(load_w),
.transpose_en(transpose_en),
.psum_out(psum_LD)
);



endmodule