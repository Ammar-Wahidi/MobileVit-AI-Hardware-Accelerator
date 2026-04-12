`timescale 1ns/1ps

module Batch_Normalization_tb;

    parameter integer Data_Width = 16;
    parameter integer FRAC_BITS  = 8;
    parameter integer N          = 16;
    parameter CLK_PERIOD         = 10;

    reg CLK, RST, INBatch_Valid;
    reg signed [Data_Width-1:0] in_row [0:N-1];
    reg signed [Data_Width-1:0] A [0:319];
    reg signed [Data_Width-1:0] B [0:319];

    wire signed [Data_Width-1:0] out_row [0:N-1];
    wire OutBatch_Valid;

    reg signed [Data_Width-1:0] x_ref [0:N-1];
    reg signed [Data_Width-1:0] y_ref [0:N-1];
    
    integer i, batch, errors = 0, tb_offset = 0;

    Batch_Normalization #(.Data_Width(Data_Width), .FRAC_BITS(FRAC_BITS), .N(N)) 
    DUT (.CLK(CLK), .RST(RST), .in_row(in_row), .INBatch_Valid(INBatch_Valid), .A(A), .B(B), .out_row(out_row), .OutBatch_Valid(OutBatch_Valid));

    initial CLK = 0;
    always #(CLK_PERIOD/2) CLK = ~CLK;

    initial begin
        $display("--- Loading Parameters ---");
        $readmemh("A.mem", A);
        $readmemh("B.mem", B);
        
        // Safety Check: Display first values to ensure files were read
        $display("Check: A[0]=%h, B[0]=%h", A[0], B[0]);

        RST = 0; INBatch_Valid = 0;
        #(2*CLK_PERIOD);
        RST = 1;

        for (batch = 0; batch < 1000; batch = batch + 1) begin
            @(posedge CLK);
            INBatch_Valid = 1;
            for (i = 0; i < N; i = i + 1) begin
                x_ref[i] = $signed($urandom()); 
                in_row[i] = x_ref[i];
            end

            // Reference Math
            for (i = 0; i < N; i = i + 1) begin
                y_ref[i] = ( (A[i+tb_offset] * x_ref[i]) + $signed({B[i+tb_offset], {FRAC_BITS{1'b0}}}) ) >>> FRAC_BITS;
            end

            @(posedge CLK);
            INBatch_Valid = 0;

            wait (OutBatch_Valid);
            #(CLK_PERIOD/4);

            // --- THE DISPLAY TABLE ---
            $display("\nBatch #%0d | Offset: %0d", batch, tb_offset);
            $display("---------------------------------------------------------------");
            $display("| Idx | Scale(A) | Bias(B) | X_in |  Y_REF (Exp) | Y_DUT (Act) |");
            $display("---------------------------------------------------------------");
            
            for (i = 0; i < N; i = i + 1) begin
                string status;
                status = (out_row[i] === y_ref[i]) ? "OK" : "FAIL";
                
                $display("| %2d  | %8d | %7d | %4d | %12d | %11d | %s", 
                         i, A[i+tb_offset], B[i+tb_offset], x_ref[i], y_ref[i], out_row[i], status);
                
                if (out_row[i] !== y_ref[i]) errors++;
            end
            $display("---------------------------------------------------------------");

            tb_offset = (tb_offset + N >= 320) ? 0 : tb_offset + N;
            #(CLK_PERIOD);
        end

        if (errors == 0) $display("\nRESULT: SUCCESS!");
        else $display("\nRESULT: FAILED with %0d errors", errors);
        $finish;
    end
endmodule