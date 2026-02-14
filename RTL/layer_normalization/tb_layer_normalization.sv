`timescale 1ns/1ps

module tb_layer_normalization;

parameter DATA_WIDTH = 32;
parameter EMBED_DIM  = 32;
parameter NUM_TESTS  = 1000;

reg clk;
reg rst_n;
reg sum_en_1;

reg  signed [DATA_WIDTH-1:0] activation_in [0:EMBED_DIM-1];
wire signed [DATA_WIDTH-1:0] normalized_output [0:EMBED_DIM-1];
wire norm_final_out_valid;

// ------------------------------------------
// DUT
// ------------------------------------------
layer_normalization_top #(
    .DATA_WIDTH(DATA_WIDTH),
    .EMBED_DIM(EMBED_DIM)
) DUT (
    .clk(clk),
    .rst_n(rst_n),
    .sum_en_1(sum_en_1),
    .activation_in(activation_in),
    .normalized_output(normalized_output),
    .norm_final_out_valid(norm_final_out_valid)
);

// ------------------------------------------
// Clock
// ------------------------------------------
always #5 clk = ~clk;

// ------------------------------------------
// Memories
// ------------------------------------------
reg signed [DATA_WIDTH-1:0] activation_mem [0:NUM_TESTS*EMBED_DIM-1];
reg signed [DATA_WIDTH-1:0] expected_mem   [0:NUM_TESTS*EMBED_DIM-1];

integer test, i;
integer error_count;

// ------------------------------------------
// Test Procedure
// ------------------------------------------
initial begin

    clk = 0;
    rst_n = 0;
    sum_en_1 = 0;
    error_count = 0;

    // Read files
    $readmemh("activation_in.txt", activation_mem);
    $readmemh("normalized_out.txt", expected_mem);

    #20;
    rst_n = 1;

    for (test = 0; test < NUM_TESTS; test = test + 1) begin

        // Load one vector
        for (i = 0; i < EMBED_DIM; i = i + 1) begin
            activation_in[i] = activation_mem[test*EMBED_DIM + i];
        end

        // Start
        @(posedge clk);
        sum_en_1 = 1;
        @(posedge clk);
        sum_en_1 = 0;

        // Wait for valid
        wait(norm_final_out_valid);

        //$display("==============================================");
        //$display("TEST %0d", test);
        //$display("--------------- INPUT VECTOR ----------------");

        for (i = 0; i < EMBED_DIM; i = i + 1)
           // $display("IN[%0d] = %h", i, activation_in[i]);

       // $display("------------- EXPECTED OUTPUT ---------------");

        for (i = 0; i < EMBED_DIM; i = i + 1)
            //$display("EXP[%0d] = %h", i, expected_mem[test*EMBED_DIM + i]);

        //$display("-------------- DUT OUTPUT -------------------");

        // Compare
        for (i = 0; i < EMBED_DIM; i = i + 1) begin

            //$display("OUT[%0d] = %h", i, normalized_output[i]);

            if (normalized_output[i] !== expected_mem[test*EMBED_DIM + i]) begin
                //$display("❌ MISMATCH at index %0d", i);
                error_count = error_count + 1;
            end
        end

        if (error_count == 0)
           // $display("✅ TEST %0d PASSED", test);
        else
            //$display("❌ TEST %0d FAILED", test);

        @(posedge clk);
    end

    $display("==============================================");
    $display("SIMULATION FINISHED");
    //$display("TOTAL ERRORS = %0d", error_count);
    $display("==============================================");

    $stop;
end

endmodule
