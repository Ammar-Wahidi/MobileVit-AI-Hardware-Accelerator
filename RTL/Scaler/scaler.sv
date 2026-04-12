`timescale 1ns / 1ps

module fixed_point_scaler #(
    parameter int FRACT_BITS = 16
)(
    input  logic signed [31:0] z_int32,   
    input  logic signed [31:0] b_scale,   
    output logic signed [31:0] z_scaled   
);

    logic signed [63:0] z_int64;
    logic signed [63:0] b_scale64;
    logic signed [63:0] full_product;
    
    assign z_int64 = z_int32;
    assign b_scale64 = b_scale;

    always_comb begin
        // 1. 64-bit Multiplication
        full_product = z_int64 * b_scale64; 
        z_scaled = { {FRACT_BITS{full_product[63]}}, full_product[63 : FRACT_BITS] };
    end

endmodule