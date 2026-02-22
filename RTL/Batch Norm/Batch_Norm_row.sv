module Batch_Norm #(
    parameter integer Data_Width = 16,   // total bits (e.g., Q8.8 format)
    parameter integer FRAC_BITS  = 8,    // fractional bits
    parameter integer N          = 32    // number of elements per row
)(
    input  wire                    CLK,
    input  wire                    RST,
    input  wire [N*Data_Width-1:0] in_row,
    input  wire                    INBatch_Valid,
    input  wire [Data_Width-1:0]   A [0:N-1], // scale (γ)
    input  wire [Data_Width-1:0]   B [0:N-1], // bias (β)
    output reg  [N*Data_Width-1:0] out_row,
    output reg                     OutBatch_Valid
);

    // internal signals
    reg  signed [Data_Width-1:0] x [0:N-1];
    reg  signed [Data_Width-1:0] y [0:N-1];

    // expanded internal math signals
    reg signed [2*Data_Width-1:0] mult_result   [0:N-1];
    reg signed [2*Data_Width-1:0] mult_shifted  [0:N-1];
    reg signed [2*Data_Width:0]   sum_result    [0:N-1];

    integer i;

    // ==============================
    // Combinational calculation
    // ==============================
    always @(*) begin
        for (i = 0; i < N; i = i + 1) begin
            // unpack input
            x[i] = in_row[i*Data_Width +: Data_Width]; // Q8.8

            // multiply A[i] * x[i]  → 2*Data_Width bits
            // signed multiplication is synthesizable
            mult_result[i] = $signed(A[i]) * $signed(x[i]); //Q16.16


            // sign-extend B[i] to same width before addition
            sum_result[i] = mult_result[i] + 
                            $signed({{(Data_Width- FRAC_BITS ){B[i][Data_Width-1]}}, B[i],{FRAC_BITS{1'b0}}}); //Q16.16 + Q16.16 = Q17.16

            // truncate/normalize result back to Data_Width bits
            // keeps the same Q format (Qm.n)
            y[i] = sum_result[i][Data_Width + FRAC_BITS - 1 : FRAC_BITS]; //Q8.8
        end
    end

    // ==============================
    // Sequential output register
    // ==============================
    always @(posedge CLK or negedge RST) begin
        if (!RST) 
        begin
            out_row        <= '0;
            OutBatch_Valid <= 1'b0;
        end 
        else 
        begin
            if (INBatch_Valid) begin
                for (i = 0; i < N; i = i + 1)
                    out_row[i*Data_Width +: Data_Width] <= y[i];

                OutBatch_Valid <= 1'b1;
            end 
            else 
            begin
                OutBatch_Valid <= 1'b0;
            end
        end
    end

endmodule
