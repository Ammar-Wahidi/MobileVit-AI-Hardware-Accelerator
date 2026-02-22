`timescale 1ns/1ps

module Batch_Norm_tb;

parameter integer Data_Width = 16;
parameter integer FRAC_BITS  = 8;
parameter integer N          = 8;
parameter CLK_PERIOD = 10;

reg CLK;
reg RST;
reg INBatch_Valid;
reg [N*Data_Width-1:0] in_row;
reg [Data_Width-1:0]   A [0:N-1];
reg [Data_Width-1:0]   B [0:N-1];

wire [N*Data_Width-1:0] out_row;
wire OutBatch_Valid;

reg signed [Data_Width-1:0] x_ref [0:N-1];
reg signed [Data_Width-1:0] y_ref [0:N-1];
reg signed [2*Data_Width-1:0] mult_result;
reg signed [2*Data_Width:0]   sum_result;

integer i, batch;
integer errors = 0;

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

initial CLK = 0;
always #(CLK_PERIOD/2) CLK = ~CLK;

initial begin
    $display("======== Starting Batch_Norm Testbench ========");

    // Load trained parameters
    $readmemh("A.mem", A);
    $readmemh("B.mem", B);

    RST = 0;
    INBatch_Valid = 0;
    in_row = 0;
    #(2*CLK_PERIOD);
    RST = 1;

    for (batch = 0; batch < 1000; batch = batch + 1) begin

        @(posedge CLK);
        INBatch_Valid = 1;

        for (i = 0; i < N; i = i + 1) begin
            x_ref[i] = $urandom_range(-256, 255);  // Q8.8 range
            in_row[i*Data_Width +: Data_Width] = x_ref[i];
        end

        @(posedge CLK);
        INBatch_Valid = 0;

        wait (OutBatch_Valid);
        #1;

        // Reference calculation
        for (i = 0; i < N; i = i + 1) begin
            mult_result = $signed(A[i]) * $signed(x_ref[i]);
            sum_result  = mult_result +
                          $signed({{(Data_Width-FRAC_BITS){B[i][Data_Width-1]}},
                           B[i],
                           {FRAC_BITS{1'b0}}});

            y_ref[i] = sum_result[Data_Width + FRAC_BITS - 1 : FRAC_BITS];
        end

        // Display
        $display("---------------------------------------------------");
        $display("Batch #%0d Results:", batch);
        $display("| Index |     X_in   |    Y_DUT   |    Y_REF   | Status |");

        for (i = 0; i < N; i = i + 1) begin
            if ($signed(out_row[i*Data_Width +: Data_Width]) !== y_ref[i]) begin
                $display("| %0d | %7d | %7d | %7d | FAIL",
                    i,
                    x_ref[i],
                    $signed(out_row[i*Data_Width +: Data_Width]),
                    y_ref[i]);
                errors = errors + 1;
            end
            else begin
                $display("| %0d | %7d | %7d | %7d | OK",
                    i,
                    x_ref[i],
                    $signed(out_row[i*Data_Width +: Data_Width]),
                    y_ref[i]);
            end
        end

        $display("---------------------------------------------------\n");
    end

    if (errors == 0)
        $display("All tests PASSED with no mismatches!");
    else
        $display("Test FAILED with %0d mismatches", errors);

    $finish;
end

endmodule
