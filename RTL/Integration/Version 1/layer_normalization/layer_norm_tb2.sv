`timescale 1ns/1ps

module layer_norm_tb2;

    parameter N = 8;
    parameter DATA_WIDTH = 8;

    // DUT connections
    logic signed [DATA_WIDTH-1:0] in_vec [N];
    logic signed [DATA_WIDTH-1:0] out_vec [N];

    // Instantiate DUT
    layer_normalization_top #(
        .N(N),
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .in_vec(in_vec),
        .out_vec(out_vec)
    );

    // Test vectors
    logic signed [DATA_WIDTH-1:0] vec1 [N] = '{5, 3, 3, 4, -1, 0, 2, 9};
    logic signed [DATA_WIDTH-1:0] vec2 [N] = '{50, 50, 50, 50, 50, 50, 50, 50};

    // Task to run a test
    task run_test(input string name, input logic signed [DATA_WIDTH-1:0] vec [N]);
        int sum, mean, variance, std, tmp;
        int ref_out [N];
        begin
            $display("\n==============================================");
            $display("==== Running Test Case: %s ====", name);
            $display("==============================================");

            // Apply input
            for (int i=0; i<N; i++) in_vec[i] = vec[i];
            #1;

            // Reference mean
            sum = 0;
            for (int i=0; i<N; i++) sum += vec[i];
            mean = sum / N;

            // Reference variance
            variance = 0;
            for (int i=0; i<N; i++) begin
                tmp = vec[i] - mean;
                variance += tmp * tmp;
            end
            variance = variance / N;

            // Reference stddev (simple sqrt)
            std = (variance == 0) ? 0 : $rtoi($sqrt(real'(variance)));

            // Reference normalized outputs
            for (int i=0; i<N; i++) begin
                if (std == 0) ref_out[i] = 0;
                else ref_out[i] = ((vec[i] - mean) * 128) / std;
            end

            // Print results
            $write(">> Input Vector: ");
            for (int i=0; i<N; i++) $write("%0d ", vec[i]);
            $display("");

            $display(">> Reference: mean=%0d, variance=%0d, std=%0d", mean, variance, std);

            $display("\n>> Normalized Outputs (DUT vs Ref)");
            for (int i=0; i<N; i++) begin
                $display("Index %0d: DUT=%0d  || Ref=%0d", i, out_vec[i], ref_out[i]);
            end
        end
    endtask

    initial begin
        $display("==== LayerNorm Testbench Start ====");
        run_test("Mixed values vector", vec1);
        run_test("Constant vector", vec2);
        $display("\n==== LayerNorm Testbench End ====");
        $finish;
    end

endmodule
