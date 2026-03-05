`timescale 1ns/1ps

module Lego_SA_BN_Wrapper #(
    parameter DATA_W       = 8,   // Activation/Weight bit-width
    parameter DATA_W_OUT   = 32,  // SA Accumulator bit-width
    parameter Y_INPUT_SIZE = 8,
    parameter FRAC_BITS    = 8,   // Fractional bits for BN math
    parameter N_TILE       = 16   // 16x16 Base Tile
)(
    input  logic                      clk,
    input  logic                      rst_n,

    // Lego_SA Top-Level Control Inputs
    input  logic                      valid_in,
    input  logic [1:0]                lego_type,
    input  logic [Y_INPUT_SIZE-1:0]   y_input_size,
    input  logic                      transpose_en,
    
    // 64-wide Data Inputs (4 * N_TILE)
    input  logic [DATA_W-1:0]         act_in       [4*N_TILE],
    input  logic [DATA_W-1:0]         weight_in    [4*N_TILE],

    // Batch Normalization Parameters (Scale/Gamma and Bias/Beta)
    input  logic [DATA_W_OUT-1:0]     bn_scale     [0:319], 
    input  logic [DATA_W_OUT-1:0]     bn_bias      [0:319],

    // Final Normalized Output (64-wide)
    output logic [DATA_W_OUT-1:0]     bn_out_row   [0:4*N_TILE-1],
    output logic                      bn_valid_out,
    output logic                      busy
);

    // Internal connection: SA output to BN input
    logic [DATA_W_OUT-1:0] sa_psum [4*N_TILE];
    logic                  sa_valid_out;

    // ----------------------------------------------------------------
    // 1. Full 4-Tile Lego Systolic Array (Includes Lego_CU)
    // ----------------------------------------------------------------
    Lego_SA #(
        .DATA_W      (DATA_W),
        .DATA_W_OUT  (DATA_W_OUT),
        .Y_INPUT_SIZE(Y_INPUT_SIZE),
        .N_TILE      (N_TILE)
    ) u_Lego_SA (
        .clk         (clk),
        .rst_n       (rst_n),
        .valid_in    (valid_in),
        .lego_type   (lego_type),
        .y_input_size(y_input_size),
        .transpose_en(transpose_en),
        .act_in      (act_in),
        .weight_in   (weight_in),
        .psum_out    (sa_psum),
        .valid_out   (sa_valid_out), // Driven automatically by Lego_CU
        .busy        (busy)
    );

    // ----------------------------------------------------------------
    // 2. Massive 64-element Batch Normalization Module
    // ----------------------------------------------------------------
    Batch_Normalization #(
        .Data_Width(DATA_W_OUT), 
        .FRAC_BITS (FRAC_BITS),
        .N         (4*N_TILE) // Expanded to 64 to handle all 4 tiles
    ) u_BN (
        .CLK            (clk),
        .RST            (rst_n),
        .in_row         (sa_psum),
        .INBatch_Valid  (sa_valid_out), // Triggered by Lego_CU
        .A              (bn_scale),
        .B              (bn_bias),
        .out_row        (bn_out_row),
        .OutBatch_Valid (bn_valid_out)
    );

endmodule