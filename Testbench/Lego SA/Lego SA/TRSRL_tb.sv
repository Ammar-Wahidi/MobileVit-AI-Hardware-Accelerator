`timescale 1ns/1ps

// ===========================================================
//  TRSRL Testbench  –  10 known + 1000 randomized test cases
//
//  Module behaviour (identical structure to TRSDL but for
//  activations shifting right instead of psums shifting down):
//    act_out[k]  =  act_in[k]  delayed by k clock cycles.
//    Hold input row STABLE for N-1=15 cycles → all 16 outputs
//    simultaneously equal the input row.
// ===========================================================

module TRSRL_tb;

parameter DATAWIDTH = 8;
parameter N_SIZE    = 16;
parameter DELAY     = N_SIZE - 1;   // 15 cycles

logic                 clk;
logic                 rst_n;
logic [DATAWIDTH-1:0] act_in  [N_SIZE];
logic [DATAWIDTH-1:0] act_out [N_SIZE];

TRSRL #(.DATAWIDTH(DATAWIDTH), .N_SIZE(N_SIZE)) dut (
    .clk    (clk),
    .rst_n  (rst_n),
    .act_in (act_in),
    .act_out(act_out)
);

initial clk = 0;
always #5 clk = ~clk;

int pass_count = 0;
int fail_count = 0;

task automatic tick; @(posedge clk); #1; endtask

task automatic drive_zero;
    for (int k = 0; k < N_SIZE; k++) act_in[k] = '0;
endtask

// ------------------------------------------------------------
// send_and_expect:
//   Hold the input row stable for DELAY cycles, then sample
//   all outputs and compare to the original input row.
// ------------------------------------------------------------
task automatic send_and_expect(
    input logic [DATAWIDTH-1:0] test_row [N_SIZE],
    input string                test_name
);
    logic ok;
    for (int k = 0; k < N_SIZE; k++) act_in[k] = test_row[k];
    repeat(DELAY) tick;

    ok = 1;
    for (int k = 0; k < N_SIZE; k++) begin
        if (act_out[k] !== test_row[k]) begin
            $display("    [%s] lane%0d: exp=0x%02h got=0x%02h",
                     test_name, k, test_row[k], act_out[k]);
            ok = 0;
        end
    end

    if (ok) begin $display("  PASS [%s]", test_name); pass_count++; end
    else    begin $display("  FAIL [%s]", test_name); fail_count++; end

    drive_zero();
    repeat(DELAY) tick;
endtask

// ===========================================================
initial begin
    $dumpfile("TRSRL_tb.vcd");
    $dumpvars(0, TRSRL_tb);

    rst_n = 0; drive_zero();
    repeat(4) tick;
    rst_n = 1; tick;

    // --------------------------------------------------------
    // TC1: All zeros
    //   in  = [0x00 x16]
    //   out = [0x00 x16]
    // --------------------------------------------------------
    $display("\nTC1: All zeros");
    begin
        logic [DATAWIDTH-1:0] row [N_SIZE];
        for (int k=0;k<N_SIZE;k++) row[k] = 8'h00;
        send_and_expect(row, "TC1_all_zeros");
    end

    // --------------------------------------------------------
    // TC2: All 0xFF (max 8-bit value)
    //   in  = [0xFF x16]
    //   out = [0xFF x16]
    // --------------------------------------------------------
    $display("\nTC2: All max 0xFF");
    begin
        logic [DATAWIDTH-1:0] row [N_SIZE];
        for (int k=0;k<N_SIZE;k++) row[k] = 8'hFF;
        send_and_expect(row, "TC2_all_max");
    end

    // --------------------------------------------------------
    // TC3: Sequential  lane[k] = k
    //   in  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    //   out = same
    // --------------------------------------------------------
    $display("\nTC3: Sequential lane[k]=k");
    begin
        logic [DATAWIDTH-1:0] row [N_SIZE];
        for (int k=0;k<N_SIZE;k++) row[k] = 8'(k);
        send_and_expect(row, "TC3_sequential");
    end

    // --------------------------------------------------------
    // TC4: Descending  lane[k] = 15-k
    //   in  = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    //   out = same
    // --------------------------------------------------------
    $display("\nTC4: Descending lane[k]=15-k");
    begin
        logic [DATAWIDTH-1:0] row [N_SIZE];
        for (int k=0;k<N_SIZE;k++) row[k] = 8'(N_SIZE-1-k);
        send_and_expect(row, "TC4_descending");
    end

    // --------------------------------------------------------
    // TC5: Alternating 0x00 / 0xFF
    //   in  = [0x00, 0xFF, 0x00, 0xFF, ...]
    //   out = same
    // --------------------------------------------------------
    $display("\nTC5: Alternating 0x00 / 0xFF");
    begin
        logic [DATAWIDTH-1:0] row [N_SIZE];
        for (int k=0;k<N_SIZE;k++) row[k] = (k%2==0) ? 8'h00 : 8'hFF;
        send_and_expect(row, "TC5_alternating");
    end

    // --------------------------------------------------------
    // TC6: Powers of 2  lane[k] = 1 << (k % 8)
    //   in  = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128]
    //   out = same
    // --------------------------------------------------------
    $display("\nTC6: Powers of 2 lane[k]=1<<(k mod 8)");
    begin
        logic [DATAWIDTH-1:0] row [N_SIZE];
        for (int k=0;k<N_SIZE;k++) row[k] = 8'(1 << (k % 8));
        send_and_expect(row, "TC6_powers_of_2");
    end

    // --------------------------------------------------------
    // TC7: Checkerboard  0xAA / 0x55
    //   in  = [0xAA, 0x55, 0xAA, 0x55, ...]
    //   out = same
    // --------------------------------------------------------
    $display("\nTC7: Checkerboard 0xAA / 0x55");
    begin
        logic [DATAWIDTH-1:0] row [N_SIZE];
        for (int k=0;k<N_SIZE;k++) row[k] = (k%2==0) ? 8'hAA : 8'h55;
        send_and_expect(row, "TC7_checkerboard");
    end

    // --------------------------------------------------------
    // TC8: Unique known value per lane
    //   lane[ 0]=0xA0   lane[ 1]=0xB1   lane[ 2]=0xC2   lane[ 3]=0xD3
    //   lane[ 4]=0xE4   lane[ 5]=0xF5   lane[ 6]=0x06   lane[ 7]=0x17
    //   lane[ 8]=0x28   lane[ 9]=0x39   lane[10]=0x4A   lane[11]=0x5B
    //   lane[12]=0x6C   lane[13]=0x7D   lane[14]=0x8E   lane[15]=0x9F
    // --------------------------------------------------------
    $display("\nTC8: Unique known value per lane");
    begin
        logic [DATAWIDTH-1:0] row [N_SIZE];
        row[ 0]=8'hA0; row[ 1]=8'hB1; row[ 2]=8'hC2; row[ 3]=8'hD3;
        row[ 4]=8'hE4; row[ 5]=8'hF5; row[ 6]=8'h06; row[ 7]=8'h17;
        row[ 8]=8'h28; row[ 9]=8'h39; row[10]=8'h4A; row[11]=8'h5B;
        row[12]=8'h6C; row[13]=8'h7D; row[14]=8'h8E; row[15]=8'h9F;
        send_and_expect(row, "TC8_unique_per_lane");
    end

    // --------------------------------------------------------
    // TC9: Single hot — only lane 7 non-zero
    //   in  = [0,0,0,0,0,0,0, 0xAB, 0,0,0,0,0,0,0,0]
    //   out = same
    // --------------------------------------------------------
    $display("\nTC9: Single hot lane7=0xAB");
    begin
        logic [DATAWIDTH-1:0] row [N_SIZE];
        for (int k=0;k<N_SIZE;k++) row[k] = 8'h00;
        row[7] = 8'hAB;
        send_and_expect(row, "TC9_single_hot_lane7");
    end

    // --------------------------------------------------------
    // TC10: Activation-style values  lane[k] = (k+1)*15
    //   in  = [15, 30, 45, 60, 75, 90, 105, 120,
    //          135, 150, 165, 180, 195, 210, 225, 240]
    //   out = same
    // --------------------------------------------------------
    $display("\nTC10: Activation style lane[k]=(k+1)*15");
    begin
        logic [DATAWIDTH-1:0] row [N_SIZE];
        for (int k=0;k<N_SIZE;k++) row[k] = 8'((k+1)*15);
        send_and_expect(row, "TC10_activation_values");
    end

    // --------------------------------------------------------
    // RANDOMIZED TESTS  TC11 – TC1010  (1000 cases)
    // --------------------------------------------------------
    $display("\n--- Randomized Tests (TC11 .. TC1010) ---");
    begin
        logic [DATAWIDTH-1:0] row [N_SIZE];
        logic [7:0]           rand_val;
        int                   rand_ok;
        int                   rand_pass, rand_fail;
        rand_pass = 0; rand_fail = 0;

        for (int tc = 11; tc <= 1010; tc++) begin

            // Build row using varied random patterns
            case ($urandom_range(0, 6))

                0: begin
                    // Fully random per lane
                    for (int k=0;k<N_SIZE;k++) row[k] = $urandom();
                end

                1: begin
                    // Same random value broadcast to all lanes
                    rand_val = $urandom();
                    for (int k=0;k<N_SIZE;k++) row[k] = rand_val;
                end

                2: begin
                    // Random base + lane offset (masked to 8-bit)
                    rand_val = $urandom();
                    for (int k=0;k<N_SIZE;k++) row[k] = 8'(rand_val + k);
                end

                3: begin
                    // Alternating between two random values
                    logic [7:0] a, b;
                    a = $urandom(); b = $urandom();
                    for (int k=0;k<N_SIZE;k++) row[k] = (k%2==0) ? a : b;
                end

                4: begin
                    // Single random hot lane, rest zero
                    int hot;
                    hot = $urandom_range(0, N_SIZE-1);
                    for (int k=0;k<N_SIZE;k++) row[k] = 8'h00;
                    row[hot] = $urandom();
                end

                5: begin
                    // Bit-rotation of random seed by lane index
                    rand_val = $urandom();
                    for (int k=0;k<N_SIZE;k++)
                        row[k] = (rand_val << (k%8)) | (rand_val >> (8-(k%8)));
                end

                6: begin
                    // Corner: mix of 0x00 and 0xFF with random positions
                    for (int k=0;k<N_SIZE;k++)
                        row[k] = ($urandom_range(0,1)) ? 8'hFF : 8'h00;
                end

            endcase

            // Drive and wait
            for (int k = 0; k < N_SIZE; k++) act_in[k] = row[k];
            repeat(DELAY) tick;

            // Check
            rand_ok = 1;
            for (int k = 0; k < N_SIZE; k++) begin
                if (act_out[k] !== row[k]) begin
                    $display("  [TC%0d] lane%0d: exp=0x%02h got=0x%02h",
                             tc, k, row[k], act_out[k]);
                    rand_ok = 0;
                end
            end

            if (rand_ok) begin pass_count++; rand_pass++; end
            else         begin fail_count++; rand_fail++;
                              $display("  FAIL [TC%0d]", tc); end

            // Flush
            drive_zero();
            repeat(DELAY) tick;
        end

        $display("  Randomized: %0d PASS  %0d FAIL (out of 1000)", rand_pass, rand_fail);
    end

    // --------------------------------------------------------
    // Summary
    // --------------------------------------------------------
    $display("\n================================================");
    $display("  TOTAL: %0d PASS   %0d FAIL", pass_count, fail_count);
    $display("================================================");
    if (fail_count == 0) $display("  >>> ALL TESTS PASSED <<<");
    else                 $display("  >>> FAILURES DETECTED <<<");

    $finish;
end

initial begin #5_000_000; $display("TIMEOUT"); $finish; end

endmodule