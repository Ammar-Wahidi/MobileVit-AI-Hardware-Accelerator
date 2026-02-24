`timescale 1ns/1ps

module tb_swish();

    localparam int WIDTH = 16;
    localparam int FRACT_BITS = 8; // Q8.8 Format

    // DUT signals
    reg  signed [WIDTH-1:0] x;
    wire signed [WIDTH-1:0] y_hw;

    // Instantiate DUT
    swish #(
        .WIDTH(WIDTH),
        .FRACT_BITS(FRACT_BITS)
    ) dut (
        .x(x),
        .y(y_hw)
    );

    // -----------------------------------------------------------
    // Function 1: Ideal Swish (SiLU) 
    // -----------------------------------------------------------
    function real calc_ideal_swish;
        input integer xi;
        real xf;
        begin
            xf = xi;                            
            calc_ideal_swish = xf / (1.0 + $exp(-xf)); 
        end
    endfunction

    // -----------------------------------------------------------
    // Function 2: Ideal H-Swish - The Mathematical Approx
    // -----------------------------------------------------------
    function real calc_ideal_h_swish;
        input integer xi;
        real xf, relu_val;
        begin
            xf = xi;
            relu_val = xf + 3.0;
            if (relu_val < 0) relu_val = 0;
            else if (relu_val > 6.0) relu_val = 6.0;
            
            calc_ideal_h_swish = xf * (relu_val / 6.0);
        end
    endfunction

    // -----------------------------------------------------------
    // Test Sequence
    // -----------------------------------------------------------
    integer i;
    real val_swish;    
    real val_h_math;   
    real val_hw_real;  
    real val_diff;     

    initial begin
        $display("\n--------------------------------------------------------------------------------------");
        $display("|  In  |   Ideal Swish    |  Ideal H-Swish   |   HW Output    |   Diff (Ideal-HW)  |");
        $display("--------------------------------------------------------------------------------------");

        for (i = -8; i <= 124; i = i + 1) begin 
            // 1. CONVERT INPUT TO Q-FORMAT (Shift left by FRACT_BITS)
            x = i * (1 << FRACT_BITS); 
            
            #1;  // Wait for hardware combinational logic
            
            val_swish  = calc_ideal_swish(i);
            val_h_math = calc_ideal_h_swish(i);
            
            // 2. CONVERT OUTPUT FROM Q-FORMAT TO REAL DECIMAL
            // Note: If real'(y_hw) gives a syntax error, use: $itor(y_hw) / (2.0 ** FRACT_BITS)
            val_hw_real = real'(y_hw) / (2.0 ** FRACT_BITS);
            
            val_diff   = val_swish - val_hw_real;

            // Print Row
            $display("| %4d | %16.6f | %16.6f | %14.6f | %18.6f |", 
                     i, val_swish, val_h_math, val_hw_real, val_diff);
        end
        $display("--------------------------------------------------------------------------------------\n");
        $finish;
    end

endmodule