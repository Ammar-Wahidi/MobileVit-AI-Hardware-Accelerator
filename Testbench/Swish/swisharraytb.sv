`timescale 1ns/1ps

module tb_swish_array();

    
    localparam int N = 4;              
    localparam int WIDTH = 16;         // 16-bit data width per element
    localparam int FRACT_BITS = 8;     // Q8.8 Format

    
    reg  signed [WIDTH-1:0] x_array [0:N-1];
    wire signed [WIDTH-1:0] y_array [0:N-1];

    swish_array #(
        .N(N),
        .WIDTH(WIDTH),
        .FRACT_BITS(FRACT_BITS)
    ) dut (
        .x(x_array),
        .y(y_array)
    );

    // -----------------------------------------------------------
    // Function 1: Ideal Swish (SiLU) 
    // -----------------------------------------------------------
    function real calc_ideal_swish(input integer xi);
        real xf;
        begin
            xf = xi;                            
            calc_ideal_swish = xf / (1.0 + $exp(-xf)); 
        end
    endfunction

    // -----------------------------------------------------------
    // Function 2: Ideal H-Swish - The Mathematical Approx
    // -----------------------------------------------------------
    function real calc_ideal_h_swish(input integer xi);
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
    integer start_val;
    integer j, current_in;
    
    real val_swish;    
    real val_h_math;   
    real val_hw_real;  
    real val_diff;     

    initial begin
        $display("\n------------------------------------------------------------------------------------------------");
        $display("| Lane |   In   |   Ideal Swish    |  Ideal H-Swish   |   HW Output    |   Diff (Ideal-HW)  |");
        $display("------------------------------------------------------------------------------------------------");

        for (start_val = -8; start_val <= 124; start_val = start_val + N) begin 
            
            // 1. LOAD THE UNPACKED ARRAY
            for (j = 0; j < N; j = j + 1) begin
                current_in = start_val + j;
                // Scale integer to Q8.8 fixed-point format
                x_array[j] = current_in * (1 << FRACT_BITS); 
            end
            
            #1;  // Wait for the hardware array to process everything
            
            // 2. READ AND PRINT THE ARRAY
            for (j = 0; j < N; j = j + 1) begin
                current_in = start_val + j;
                
                val_swish  = calc_ideal_swish(current_in);
                val_h_math = calc_ideal_h_swish(current_in);
                
                // Extract real decimal from hardware's Q-format output on lane 'j'
                val_hw_real = real'(y_array[j]) / (2.0 ** FRACT_BITS);
                val_diff    = val_swish - val_hw_real;

                // Print Row (Includes the Lane Index 'j')
                $display("|  [%0d] | %6d | %16.6f | %16.6f | %14.6f | %18.6f |", 
                         j, current_in, val_swish, val_h_math, val_hw_real, val_diff);
            end
            
            // Print a separator between parallel batches for readability
            $display("------------------------------------------------------------------------------------------------");
        end
        $finish;
    end

endmodule