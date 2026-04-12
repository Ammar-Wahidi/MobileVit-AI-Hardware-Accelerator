module Batch_Normalization #(
    parameter integer Data_Width = 16,   // total bits (e.g., Q8.8 format)
    parameter integer FRAC_BITS  = 8,    // fractional bits
    parameter integer N          = 32    // number of elements per row
)(
    input  wire                          CLK,
    input  wire                          RST,
    input  wire signed [Data_Width-1:0]  in_row [0:N-1],
    input  wire                          INBatch_Valid,
    input  wire signed [Data_Width-1:0]  A [0:319], // scale (γ) Q8.8
    input  wire signed [Data_Width-1:0]  B [0:319], // bias (β)  Q8.8
    output reg  signed [Data_Width-1:0]  out_row [0:N-1],
    output reg                           OutBatch_Valid
);

    
    reg  signed [Data_Width-1:0] x [0:N-1];
    reg  signed [Data_Width-1:0] y [0:N-1];

    reg  signed [2*Data_Width-1:0] mult_result [0:N-1];
    reg  signed [2*Data_Width-1:0] mult_shifted [0:N-1]; 
    reg  signed [2*Data_Width:0]   sum_result  [0:N-1];

    integer i;

    // =====================================================
    // Channel offset counter
    // =====================================================
    reg [9:0] channel_base;   // enough for 320

    always @(posedge CLK or negedge RST) 
    begin
        if (!RST)
            channel_base <= 0;
        else if (INBatch_Valid) begin
            if (channel_base + N >= 320)
                channel_base <= 0;
            else
                channel_base <= channel_base + N;
        end
    end
    // ==============================
    // Combinational calculation
    // ==============================
    always @(*) begin
        for (i = 0; i < N; i = i + 1) begin
            
            x[i] = in_row[i]; // Q8.8

            // multiply A[i] * x[i]  → 2*Data_Width bits
            // signed multiplication is synthesizable
            mult_result[i] = (A[i+channel_base]) * (x[i]); //Q16.16


            // sign-extend B[i] to same width before addition
            sum_result[i] = mult_result[i] + 
                            ({{(Data_Width- FRAC_BITS ){B[i+channel_base][Data_Width-1]}}, B[i+channel_base],{FRAC_BITS{1'b0}}}); //Q16.16 + Q16.16 = Q17.16

            // truncate/normalize result back to Data_Width bits
            // keeps the same Q format (Qm.n)
            y[i] = sum_result[i][Data_Width + FRAC_BITS - 1 : FRAC_BITS]; //Q8.8 from 8 to 23 >> 16 bit (Remove padding we do)
        end
    end

    // ==============================
    // Sequential output register
    // ==============================
    always @(posedge CLK or negedge RST) begin
        if (!RST) 
        begin
            OutBatch_Valid <= 1'b0;
            for (i = 0; i < N; i = i + 1)
                    out_row[i] <= 0;
        end 
        else 
        begin
            if (INBatch_Valid) begin
                for (i = 0; i < N; i = i + 1)
                    out_row[i] <= y[i];

                OutBatch_Valid <= 1'b1;
            end 
            else 
            begin
                OutBatch_Valid <= 1'b0;
            end
        end
    end

endmodule
