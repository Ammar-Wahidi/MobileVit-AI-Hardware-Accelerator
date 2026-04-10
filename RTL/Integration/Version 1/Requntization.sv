`timescale 1ns/1ps

module requantize_unit #(
    parameter int ACC_W   = 32,  // Input accumulator width
    parameter int OUT_W   = 8,   // Output quantized width
    parameter int SCALE_W = 16   // Scale multiplier width
)(
    input  logic                 clk,
    input  logic                 rst_n,
    input  logic                 valid_in,
    
    // Data Inputs
    input  logic signed [ACC_W-1:0]   acc_val,
    input  logic signed [SCALE_W-1:0] scale_mult, // M0 multiplier
    input  logic [4:0]                shift_val,  // Right shift amount
    
    // Outputs
    output logic signed [OUT_W-1:0]   req_out,
    output logic                      valid_out
);

    // Internal signals
    logic signed [ACC_W+SCALE_W-1:0] mul_res;
    logic signed [ACC_W+SCALE_W-1:0] shifted_res;
    
    // Maximum and Minimum values for clipping (e.g., 8-bit signed: 127 to -128)
    localparam MAX_VAL = (1 << (OUT_W - 1)) - 1;
    localparam MIN_VAL = -(1 << (OUT_W - 1));

    // Pipeline registers
    logic signed [ACC_W-1:0] acc_reg;
    logic valid_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_reg   <= '0;
            req_out   <= '0;
            valid_q   <= 1'b0;
            valid_out <= 1'b0;
        end else begin
            // Stage 1: Register inputs
            acc_reg <= acc_val;
            valid_q <= valid_in;
            
            // Stage 2: Math and Clipping (Can be split into 2 stages if timing fails)
            if (shifted_res > MAX_VAL)
                req_out <= MAX_VAL;
            else if (shifted_res < MIN_VAL)
                req_out <= MIN_VAL;
            else
                req_out <= shifted_res[OUT_W-1:0];
                
            valid_out <= valid_q;
        end
    end

    // Combinational Math
    always_comb begin
        // 1. Multiply by scale
        mul_res = acc_reg * scale_mult;
        
        // 2. Right shift (with rounding trick: add half of shift value before shifting)
        // Note: For strict hardware efficiency, you can drop the rounding addition.
        shifted_res = (mul_res + (1 << (shift_val - 1))) >>> shift_val; 
    end

endmodule
