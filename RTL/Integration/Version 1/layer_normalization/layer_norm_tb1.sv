`timescale 1ns/1ps

module layer_norm_tb1;

    parameter DATA_WIDTH = 16;
    parameter EMBED_DIM  = 8;

    reg clk;
    reg rst_n;
    reg layernorm_start;
    reg signed [DATA_WIDTH-1:0] activation_in [0:EMBED_DIM-1];
    wire layernorm_done;
    wire signed [DATA_WIDTH-1:0] normalized_out [0:EMBED_DIM-1];

    // DUT instantiation
    layer_normalization_top #(
        .DATA_WIDTH(DATA_WIDTH),
        .EMBED_DIM (EMBED_DIM)
    ) DUT (
        .clk(clk),
        .rst_n(rst_n),
        .layernorm_start(layernorm_start),
        .activation_in(activation_in),
        .layernorm_done(layernorm_done),
        .normalized_out(normalized_out)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100 MHz clock
    end

    // Newton-Raphson sqrt (same as DUT!)
    function integer int_sqrt_hwstyle(input integer variance);
        integer std, k;
        begin
            std = variance >>> 1; // initial guess
            for (k = 0; k < 4; k++) begin
                if (std != 0)
                    std = (std + variance/std) >>> 1;
            end
            return std;
        end
    endfunction

    // Task to run one test
    task run_test(input string name);
        integer sum, mean, sum_sq, variance, std;
        integer ref_out [0:EMBED_DIM-1];
        integer i, tmp;
        begin
            $display("\n==============================================");
            $display("==== Running Test Case: %s ====", name);
            $display("==============================================");

            // Start DUT
            #10 layernorm_start = 1;
            #10 layernorm_start = 0;
            wait(layernorm_done);
            #10;

            // Compute mean
            sum = 0;
            for (i=0; i<EMBED_DIM; i++) sum += activation_in[i];
            mean = sum >>> $clog2(EMBED_DIM);

            // Compute variance
            sum_sq = 0;
            for (i=0; i<EMBED_DIM; i++) begin
                tmp = activation_in[i] - mean;
                sum_sq += tmp * tmp;
            end
            variance = sum_sq >>> $clog2(EMBED_DIM);

            // Compute std_dev using hardware-style Newton
            std = int_sqrt_hwstyle(variance);

            // Compute normalized ref outputs
            for (i=0; i<EMBED_DIM; i++) begin
                if (std != 0)
                    ref_out[i] = (activation_in[i] - mean) / std;
                else
                    ref_out[i] = 0;
            end

            // Display inputs
            $write(">> Input Vector: ");
            for (i=0; i<EMBED_DIM; i++) $write("%0d ", activation_in[i]);
            $display("");

            // Display stats
            $display(">> LayerNorm Reference Values (Integer)");
            $display("Mean = %0d, Variance = %0d, StdDev = %0d", mean, variance, std);

            // Compare outputs
            $display("\n>> Normalized Outputs (DUT vs Reference)");
            for (i=0; i<EMBED_DIM; i++) begin
                $display("Index %0d: DUT = %0d  || Ref = %0d", 
                          i, normalized_out[i], ref_out[i]);
            end
        end
    endtask

    // Test procedure
    initial begin
        $display("==== LayerNorm Testbench Start ====");
        rst_n = 0;
        layernorm_start = 0;
        #20 rst_n = 1;

        // Test case 1: mixed values
        activation_in = '{5, 3, 3, 4, -1, 0, 2, 9};
        run_test("Mixed values vector");

        // Test case 2: constant values
        activation_in = '{50, 50, 50, 50, 50, 50, 50, 50};
        run_test("Constant vector");

        $display("\n==== LayerNorm Testbench End ====");
        $stop;
    end

endmodule
