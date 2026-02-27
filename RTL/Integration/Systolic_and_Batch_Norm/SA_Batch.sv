`timescale 1ns/1ps

module SA_Tile_BN_Wrapper #(
    parameter DATA_W     = 8,   // Activation/Weight bit-width
    parameter DATA_W_OUT = 32,  // SA Accumulator bit-width & BN input width
    parameter FRAC_BITS  = 16,  // Adjusted fractional bits for 32-bit BN math
    parameter N_TILE     = 16   // 16x16 Tile
)(
    input  logic                      clk,
    input  logic                      rst_n,

    // SA Tile Inputs
    input  logic [DATA_W-1:0]         act_in       [N_TILE],
    input  logic [DATA_W-1:0]         weight_in    [N_TILE],
    input  logic                      transpose_en,
    input  logic                      load_w,
    
    // Control signal from your CU (tells BN the SA output is ready)
    input  logic                      sa_valid_out,

    // Batch Normalization Parameters (Scale/Gamma and Bias/Beta)
    input  logic [DATA_W_OUT-1:0]     bn_scale     [0:319], 
    input  logic [DATA_W_OUT-1:0]     bn_bias      [0:319],

    // Final Normalized Output
    output logic [DATA_W_OUT-1:0]     bn_out_row   [0:N_TILE-1],
    output logic                      bn_valid_out
);

    // Internal connection: SA output to BN input
    logic [DATA_W_OUT-1:0] sa_psum [N_TILE];

    // ----------------------------------------------------------------
    // 1. Systolic Array Tile (Pure Datapath)
    // ----------------------------------------------------------------
    L_SA_NxN_top #(
        .DATA_W    (DATA_W),
        .DATA_W_OUT(DATA_W_OUT),
        .N_SIZE    (N_TILE)
    ) u_SA_Tile (
        .clk         (clk),
        .rst_n       (rst_n),
        .act_in      (act_in),
        .weight_in   (weight_in),
        .transpose_en(transpose_en),
        .load_w      (load_w),
        .psum_out    (sa_psum)
    );

    // ----------------------------------------------------------------
    // 2. Batch Normalization Module
    // ----------------------------------------------------------------
    Batch_Normalization #(
        .Data_Width(DATA_W_OUT), // Matched to SA output
        .FRAC_BITS (FRAC_BITS),
        .N         (N_TILE)
    ) u_BN (
        .CLK            (clk),
        .RST            (rst_n),
        .in_row         (sa_psum),
        .INBatch_Valid  (sa_valid_out), // Triggered when SA row is done
        .A              (bn_scale),
        .B              (bn_bias),
        .out_row        (bn_out_row),
        .OutBatch_Valid (bn_valid_out)
    );

endmodule