`timescale 1ns / 1ps

module fixed_point_requantizer #(
    parameter int FRACT_BITS = 16 
)(
    input  logic signed [31:0] q_a,    
    input  logic signed [31:0] b_mult, 
    
    
    output logic signed [7:0]  q_o     
);

    logic signed [63:0] qa_64;
    logic signed [63:0] b_mult_64;
    logic signed [63:0] mult_result;
    logic signed [63:0] concat_result;

    assign qa_64 = q_a;
    assign b_mult_64 = b_mult;

    always_comb begin
        //  64-bit Multiplication
        mult_result = qa_64 * b_mult_64;
        
        concat_result = { {FRACT_BITS{mult_result[63]}}, mult_result[63 : FRACT_BITS] };

        //  Clamping to INT8
        if (concat_result > 64'sd127) begin
            q_o = 8'sd127;
        end else if (concat_result < -64'sd128) begin
            q_o = -8'sd128;
        end else begin
            q_o = concat_result[7:0];
        end
    end

endmodule