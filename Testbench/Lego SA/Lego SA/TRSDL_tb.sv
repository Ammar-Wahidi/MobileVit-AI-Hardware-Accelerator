`timescale 1ns/1ps

// ===========================================================
//  TRSDL Testbench  –  10 known test cases
//
//  How the module works:
//    psum_out[k]  =  psum_in[k]  delayed by k clock cycles.
//    Hold input row STABLE for 15 cycles (DELAY) then read
//    all 16 outputs at once  →  they all equal the input row.
// ===========================================================

module TRSDL_tb;

parameter DATAWIDTH = 32;
parameter N_SIZE    = 16;
parameter DELAY     = N_SIZE - 1;   // 15 cycles

logic                  clk;
logic                  rst_n;
logic [DATAWIDTH-1:0]  psum_in  [N_SIZE];
logic [DATAWIDTH-1:0]  psum_out [N_SIZE];

TRSDL #(.DATAWIDTH(DATAWIDTH), .N_SIZE(N_SIZE)) dut (
    .clk(clk), .rst_n(rst_n),
    .psum_in(psum_in), .psum_out(psum_out)
);

initial clk = 0;
always #5 clk = ~clk;

int pass_count = 0;
int fail_count = 0;

task automatic tick; @(posedge clk); #1; endtask

task automatic drive_zero;
    for (int k = 0; k < N_SIZE; k++) psum_in[k] = '0;
endtask

// -------------------------------------------------------
// send_and_expect
//   1. Hold the input row stable for DELAY cycles
//   2. Sample all outputs and compare to expected
//   3. Flush with zeros before next test
// -------------------------------------------------------
task automatic send_and_expect(
    input logic [DATAWIDTH-1:0] test_row [N_SIZE],
    input string                test_name
);
    logic ok;
    for (int k = 0; k < N_SIZE; k++) psum_in[k] = test_row[k];
    repeat(DELAY) tick;

    ok = 1;
    for (int k = 0; k < N_SIZE; k++) begin
        if (psum_out[k] !== test_row[k]) begin
            $display("    [%s] lane%0d: exp=0x%08h got=0x%08h",
                     test_name, k, test_row[k], psum_out[k]);
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
    $dumpfile("TRSDL_tb.vcd");
    $dumpvars(0, TRSDL_tb);

    rst_n = 0; drive_zero();
    repeat(4) tick;
    rst_n = 1; tick;

    // -------------------------------------------------------
    // TC1  All zeros
    //   in  = [0x00000000  x16]
    //   out = [0x00000000  x16]
    // -------------------------------------------------------
    $display("\nTC1: All zeros");
    begin
        logic [DATAWIDTH-1:0] row [N_SIZE];
        for (int k=0;k<N_SIZE;k++) row[k] = 32'h0000_0000;
        send_and_expect(row, "TC1_all_zeros");
    end

    // -------------------------------------------------------
    // TC2  All 0xFFFFFFFF (max)
    //   in  = [0xFFFFFFFF  x16]
    //   out = [0xFFFFFFFF  x16]
    // -------------------------------------------------------
    $display("\nTC2: All max 0xFFFFFFFF");
    begin
        logic [DATAWIDTH-1:0] row [N_SIZE];
        for (int k=0;k<N_SIZE;k++) row[k] = 32'hFFFF_FFFF;
        send_and_expect(row, "TC2_all_max");
    end

    // -------------------------------------------------------
    // TC3  Sequential  lane[k] = k
    //   in  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    //   out = same
    // -------------------------------------------------------
    $display("\nTC3: Sequential lane[k]=k");
    begin
        logic [DATAWIDTH-1:0] row [N_SIZE];
        for (int k=0;k<N_SIZE;k++) row[k] = 32'(k);
        send_and_expect(row, "TC3_sequential");
    end

    // -------------------------------------------------------
    // TC4  Descending  lane[k] = 15-k
    //   in  = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    //   out = same
    // -------------------------------------------------------
    $display("\nTC4: Descending lane[k]=15-k");
    begin
        logic [DATAWIDTH-1:0] row [N_SIZE];
        for (int k=0;k<N_SIZE;k++) row[k] = 32'(N_SIZE-1-k);
        send_and_expect(row, "TC4_descending");
    end

    // -------------------------------------------------------
    // TC5  Alternating 0 / 0xFFFFFFFF
    //   in  = [0, FFFFFFFF, 0, FFFFFFFF, 0, FFFFFFFF, ...]
    //   out = same
    // -------------------------------------------------------
    $display("\nTC5: Alternating 0x00000000 / 0xFFFFFFFF");
    begin
        logic [DATAWIDTH-1:0] row [N_SIZE];
        for (int k=0;k<N_SIZE;k++)
            row[k] = (k%2==0) ? 32'h0000_0000 : 32'hFFFF_FFFF;
        send_and_expect(row, "TC5_alternating");
    end

    // -------------------------------------------------------
    // TC6  Powers of 2   lane[k] = 1 << k
    //   in  = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    //          1024, 2048, 4096, 8192, 16384, 32768]
    //   out = same
    // -------------------------------------------------------
    $display("\nTC6: Powers of 2 lane[k]=1<<k");
    begin
        logic [DATAWIDTH-1:0] row [N_SIZE];
        for (int k=0;k<N_SIZE;k++) row[k] = 32'(1 << k);
        send_and_expect(row, "TC6_powers_of_2");
    end

    // -------------------------------------------------------
    // TC7  Checkerboard  0xAAAAAAAA / 0x55555555
    //   in  = [AAAAAAAA, 55555555, AAAAAAAA, 55555555, ...]
    //   out = same
    // -------------------------------------------------------
    $display("\nTC7: Checkerboard 0xAAAAAAAA/0x55555555");
    begin
        logic [DATAWIDTH-1:0] row [N_SIZE];
        for (int k=0;k<N_SIZE;k++)
            row[k] = (k%2==0) ? 32'hAAAA_AAAA : 32'h5555_5555;
        send_and_expect(row, "TC7_checkerboard");
    end

    // -------------------------------------------------------
    // TC8  Unique known value per lane
    //   lane[ 0]=0xDEAD0000   lane[ 1]=0xBEEF0001
    //   lane[ 2]=0xCAFE0002   lane[ 3]=0xBABE0003
    //   lane[ 4]=0x12340004   lane[ 5]=0x56780005
    //   lane[ 6]=0x9ABC0006   lane[ 7]=0xDEF00007
    //   lane[ 8]=0x00000008   lane[ 9]=0xFFFF0009
    //   lane[10]=0xA5A5000A   lane[11]=0x5A5A000B
    //   lane[12]=0x1111000C   lane[13]=0x2222000D
    //   lane[14]=0x3333000E   lane[15]=0x4444000F
    // -------------------------------------------------------
    $display("\nTC8: Unique known value per lane");
    begin
        logic [DATAWIDTH-1:0] row [N_SIZE];
        row[ 0]=32'hDEAD_0000; row[ 1]=32'hBEEF_0001;
        row[ 2]=32'hCAFE_0002; row[ 3]=32'hBABE_0003;
        row[ 4]=32'h1234_0004; row[ 5]=32'h5678_0005;
        row[ 6]=32'h9ABC_0006; row[ 7]=32'hDEF0_0007;
        row[ 8]=32'h0000_0008; row[ 9]=32'hFFFF_0009;
        row[10]=32'hA5A5_000A; row[11]=32'h5A5A_000B;
        row[12]=32'h1111_000C; row[13]=32'h2222_000D;
        row[14]=32'h3333_000E; row[15]=32'h4444_000F;
        send_and_expect(row, "TC8_unique_per_lane");
    end

    // -------------------------------------------------------
    // TC9  Single hot — only lane 7 non-zero
    //   in  = [0,0,0,0,0,0,0, 0xABCD1234, 0,0,0,0,0,0,0,0]
    //   out = same
    // -------------------------------------------------------
    $display("\nTC9: Single hot lane7=0xABCD1234");
    begin
        logic [DATAWIDTH-1:0] row [N_SIZE];
        for (int k=0;k<N_SIZE;k++) row[k] = 32'h0;
        row[7] = 32'hABCD_1234;
        send_and_expect(row, "TC9_single_hot_lane7");
    end

    // -------------------------------------------------------
    // TC10  Partial-sum style  lane[k] = (k+1)*100
    //   in  = [100, 200, 300, 400, 500, 600, 700, 800,
    //          900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]
    //   out = same
    // -------------------------------------------------------
    $display("\nTC10: Partial-sum style lane[k]=(k+1)*100");
    begin
        logic [DATAWIDTH-1:0] row [N_SIZE];
        for (int k=0;k<N_SIZE;k++) row[k] = 32'((k+1)*100);
        send_and_expect(row, "TC10_partial_sum_values");
    end

    // -------------------------------------------------------
    // TC11 – TC1010 : 1000 randomized test cases
    // -------------------------------------------------------
    $display("\n--- Randomized Tests (TC11 .. TC1010) ---");
    begin
        logic [DATAWIDTH-1:0] row [N_SIZE];
        logic [31:0]          rand_val;
        int                   rand_ok;
        int                   rand_pass = 0;
        int                   rand_fail = 0;
        string                tname;

        for (int tc = 11; tc <= 1010; tc++) begin

            // --- Build a random row ---
            // Pick a random pattern style each iteration for variety
            rand_val = $urandom();
            case (rand_val[2:0])

                3'd0: begin
                    // Fully random per lane
                    for (int k=0;k<N_SIZE;k++) row[k] = $urandom();
                end

                3'd1: begin
                    // Same random value broadcast to all lanes
                    rand_val = $urandom();
                    for (int k=0;k<N_SIZE;k++) row[k] = rand_val;
                end

                3'd2: begin
                    // Random base + lane offset
                    rand_val = $urandom() & 32'hFFFF_FF00;
                    for (int k=0;k<N_SIZE;k++) row[k] = rand_val + k;
                end

                3'd3: begin
                    // Random base * (k+1)  — partial-sum style
                    rand_val = $urandom() & 32'h0000_FFFF;
                    for (int k=0;k<N_SIZE;k++) row[k] = rand_val * (k+1);
                end

                3'd4: begin
                    // Alternating between two random values
                    logic [31:0] a, b;
                    a = $urandom(); b = $urandom();
                    for (int k=0;k<N_SIZE;k++) row[k] = (k%2==0) ? a : b;
                end

                3'd5: begin
                    // Single random hot lane, rest zero
                    int hot;
                    hot = $urandom_range(0, N_SIZE-1);
                    for (int k=0;k<N_SIZE;k++) row[k] = 32'h0;
                    row[hot] = $urandom();
                end

                3'd6: begin
                    // Bit-shift sweep: lane[k] = rand rotated by k
                    rand_val = $urandom();
                    for (int k=0;k<N_SIZE;k++)
                        row[k] = (rand_val << k) | (rand_val >> (32-k));
                end

                default: begin
                    // Fully random per lane (fallback)
                    for (int k=0;k<N_SIZE;k++) row[k] = $urandom();
                end

            endcase

            // --- Apply and check ---
            for (int k = 0; k < N_SIZE; k++) psum_in[k] = row[k];
            repeat(DELAY) tick;

            rand_ok = 1;
            for (int k = 0; k < N_SIZE; k++) begin
                if (psum_out[k] !== row[k]) begin
                    $display("  [TC%0d] lane%0d: exp=0x%08h got=0x%08h",
                             tc, k, row[k], psum_out[k]);
                    rand_ok = 0;
                end
            end

            if (rand_ok) begin
                pass_count++; rand_pass++;
            end else begin
                fail_count++; rand_fail++;
                $display("  FAIL [TC%0d]", tc);
            end

            // Flush
            drive_zero();
            repeat(DELAY) tick;
        end

        $display("  Randomized: %0d PASS  %0d FAIL (out of 1000)", rand_pass, rand_fail);
    end

    // -------------------------------------------------------
    $display("\n================================================");
    $display("  TOTAL: %0d PASS   %0d FAIL", pass_count, fail_count);
    $display("================================================");
    if (fail_count == 0) $display("  >>> ALL TESTS PASSED <<<");
    else                 $display("  >>> FAILURES DETECTED <<<");
    $finish;
end

initial begin #5_000_000; $display("TIMEOUT"); $finish; end

endmodule