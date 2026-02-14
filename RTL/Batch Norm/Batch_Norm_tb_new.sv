`timescale 1ns/1ps

module Batch_Norm_tb;

    // -----------------------
    // Parameters
    // -----------------------
    parameter integer Data_Width = 16;
    parameter integer FRAC_BITS  = 8;  // Q8.8 fixed-point
    parameter integer N          = 8;  // small for simulation
    parameter CLK_PERIOD = 10;

    // -----------------------
    // DUT I/O
    // -----------------------
    reg                     CLK;
    reg                     RST;
    reg                     INBatch_Valid;
    reg  [N*Data_Width-1:0] in_row;
    reg  [Data_Width-1:0]   A [0:N-1];
    reg  [Data_Width-1:0]   B [0:N-1];
    wire [N*Data_Width-1:0] out_row;
    wire                    OutBatch_Valid;

    // -----------------------
    // Reference model
    // -----------------------
    reg signed [Data_Width-1:0] x_ref [0:N-1];
    reg signed [Data_Width-1:0] y_ref [0:N-1];
    reg signed [2*Data_Width-1:0] mult_result;
    reg signed [2*Data_Width-1:0] mult_shifted;
    reg signed [2*Data_Width:0]   sum_result;

    integer i, batch;
    integer errors = 0;

    // -----------------------
    // Instantiate DUT
    // -----------------------
    Batch_Norm #(
        .Data_Width(Data_Width),
        .FRAC_BITS(FRAC_BITS),
        .N(N)
    ) DUT (
        .CLK(CLK),
        .RST(RST),
        .in_row(in_row),
        .INBatch_Valid(INBatch_Valid),
        .A(A),
        .B(B),
        .out_row(out_row),
        .OutBatch_Valid(OutBatch_Valid)
    );

    // -----------------------
    // Clock generation
    // -----------------------
    initial CLK = 0;
    always #(CLK_PERIOD/2) CLK = ~CLK;

    // -----------------------
    // Stimulus
    // -----------------------
    initial begin
        $display("======== Starting Batch_Norm Testbench ========");
        RST = 0;
        INBatch_Valid = 0;
        in_row = 0;
        #(2*CLK_PERIOD);
        RST = 1;

        // ------------- Initialize A, B -------------
        for (i = 0; i < N; i = i + 1) begin
            // Example values in Q8.8 format
            // A ≈ scale, B ≈ bias
            A[i] = $rtoi(1.0  * (1 << FRAC_BITS)); // 1.0 to 256 Q8.8
            B[i] = $rtoi(0.25 * (1 << FRAC_BITS)); // 0.25 to 64 Q8.8
        end

        // ------------- Test multiple batches -------------
        for (batch = 0; batch < 5; batch = batch + 1) begin
            @(posedge CLK);
            INBatch_Valid = 1;
            // generate random input x
            for (i = 0; i < N; i = i + 1) begin
                x_ref[i] = $rtoi(($urandom_range(-100, 100) / 100.0) * (1 << FRAC_BITS));
                in_row[i*Data_Width +: Data_Width] = x_ref[i];
            end
            @(posedge CLK);
            INBatch_Valid = 0;

            // Wait for DUT output
            wait (OutBatch_Valid == 1);
            #1;

            // ------------- Compute reference -------------
            for (i = 0; i < N; i = i + 1) begin
                mult_result = $signed(A[i]) * $signed(x_ref[i]);
                //mult_shifted = mult_result >>> FRAC_BITS;
                sum_result = mult_result + {{(Data_Width- FRAC_BITS ){B[i][Data_Width-1]}}, B[i],{FRAC_BITS{1'b0}}}; //Q16.16 + Q16.16 = Q17.16
                y_ref[i] = sum_result[Data_Width + FRAC_BITS - 1 : FRAC_BITS];
            end

            // ------------- Display all results -------------
            $display("---------------------------------------------------");
            $display("Batch #%0d Results:", batch);
            $display("---------------------------------------------------");
            $display("| Index |     X_in (Q8.8)   |    Y_DUT (Q8.8)   |    Y_REF (Q8.8)   |  Status  |");
            $display("---------------------------------------------------");

            // ------------- Compare results -------------
            for (i = 0; i < N; i = i + 1) begin
                if (out_row[i*Data_Width +: Data_Width] !== y_ref[i]) begin
                    $display("|  %0d   | %7d | %7d | %7d |   Mismatch |", 
                              i, $signed(x_ref[i]), 
                              $signed(out_row[i*Data_Width +: Data_Width]), 
                              y_ref[i]);
                    errors = errors + 1;
                end else begin
                    $display("|  %0d   | %7d | %7d | %7d |   Match    |", 
                              i, $signed(x_ref[i]), 
                              $signed(out_row[i*Data_Width +: Data_Width]), 
                              y_ref[i]);
                end
            end
            $display("---------------------------------------------------\n");
        end

        // ---------------- Summary ----------------
        if (errors == 0)
            $display("All tests PASSED with no mismatches!");
        else
            $display("Test FAILED with %0d mismatches total", errors);

        $display("======== End of Simulation ========");
        $finish;
    end

endmodule
