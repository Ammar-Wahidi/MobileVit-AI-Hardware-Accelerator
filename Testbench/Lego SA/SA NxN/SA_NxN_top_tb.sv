`timescale 1ns/1ps

// ================================================================
//  SA_16x16_top Testbench — no stalls
//
//  ── Weight loading — NORMAL mode (transpose_en=0) ───────────
//
//  Weights enter from the BOTTOM edge and shift UP one row/cycle.
//  Entry:  weight_D_sig[N][col] = weight_in[col]
//
//  After N load clocks:
//    PE[row][col].W_reg = weight_in[col] that was driven at load clock (row+1)
//
//  To get PE[row][col] = W[row][col]:
//    loop k=0 → weight_in[col] = W[0][col] → arrives at PE[0] (TOP)
//    loop k=1 → weight_in[col] = W[1][col] → arrives at PE[1]
//    ...
//    loop k=N-1 → weight_in[col] = W[N-1][col] → arrives at PE[N-1] (BOTTOM)
//
//  Rule: ASCENDING row order.  W[0] first → TOP row.
//
//  Your example:
//    W = | x0 x1 x2 |   row 0 → send first → goes to PE[0] (TOP)
//        | x3 x4 x5 |   row 1 → send second
//        | x6 x7 x8 |   row 2 → send last  → goes to PE[N-1] (BOTTOM)
//
//  ── Weight loading — TRANSPOSE mode (transpose_en=1) ────────
//
//  Weights enter from the RIGHT edge and shift LEFT one col/cycle.
//  Entry:  weight_L_sig[row][N] = weight_in[row]
//
//  After N load clocks:
//    PE[row][col].W_reg = weight_in[row] that was driven at load clock (col+1)
//
//  To get PE[row][col] = W[row][col]:
//    loop k=0 → weight_in[row] = W[row][0] → arrives at PE[row][0] (LEFT)
//    loop k=1 → weight_in[row] = W[row][1] → arrives at PE[row][1]
//    ...
//    loop k=N-1 → weight_in[row] = W[row][N-1] → arrives at PE[row][N-1] (RIGHT)
//
//  Rule: ASCENDING column order.  W[*][0] first → LEFT column.
//
//  ── CU start pulse ───────────────────────────────────────────
//
//  CU starts in IDLE.  valid_in=1 moves it to LOAD_W but
//  load_w = (state==LOAD_W)&&valid_in = 0 that cycle (still IDLE).
//  Nothing is loaded.  This "start pulse" tick must happen before
//  the N weight ticks.
//
//  ── Golden model ─────────────────────────────────────────────
//  C[i][col] = SUM_row ( A[i][row] * W[row][col] )
//
//  ── Test plan ────────────────────────────────────────────────
//  TC1: Normal     A=ones    W=ones           sanity
//  TC2: Normal     A=3*I     W=(r+1)*(c+1)    row isolation
//  TC3: Normal     A=zeros   W=unique         C must be zero
//  TC4: Normal     A=unique  W=unique(prime)  full general
//  TC5: Transpose  A=3*I     W=(r+1)*(c+1)    same W as TC2 via
//                                             transpose path →
//                                             result must match TC2
// ================================================================

module SA_NxN_top_tb;

// ── Parameters ────────────────────────────────────────────────
parameter DATA_W     = 8;
parameter DATA_W_OUT = 32;
parameter N          = 16;

// ── DUT signals ───────────────────────────────────────────────
logic                   clk;
logic                   rst_n;
logic [DATA_W-1:0]      act_in    [N];
logic [DATA_W-1:0]      weight_in [N];
logic                   transpose_en;
logic                   valid_in;
logic                   valid_out;
logic                   busy;
logic [DATA_W_OUT-1:0]  psum_out  [N];

// ── DUT ───────────────────────────────────────────────────────
SA_NxN_top #(
    .DATA_W    (DATA_W    ),
    .DATA_W_OUT(DATA_W_OUT),
    .N_SIZE    (N         )
) dut (
    .clk         (clk         ),
    .rst_n       (rst_n       ),
    .act_in      (act_in      ),
    .weight_in   (weight_in   ),
    .transpose_en(transpose_en),
    .valid_in    (valid_in    ),
    .valid_out   (valid_out   ),
    .busy        (busy        ),
    .psum_out    (psum_out    )
);

// ── Clock ─────────────────────────────────────────────────────
initial clk = 0;
always  #5 clk = ~clk;

// ── Scoreboard ────────────────────────────────────────────────
int pass_count = 0;
int fail_count = 0;

// ── Helpers ───────────────────────────────────────────────────
task automatic tick; @(posedge clk); #1; endtask

task automatic zero_inputs;
    for (int k=0; k<N; k++) begin act_in[k]='0; weight_in[k]='0; end
    valid_in=0; transpose_en=0;
endtask

task automatic full_reset;
    rst_n=0; zero_inputs();
    repeat(4) tick;
    rst_n=1; tick;
endtask

task automatic NO_reset;
    rst_n=1; 
    repeat(4) tick;
    rst_n=1; tick;
endtask

// ================================================================
//  run_matmul
//
//  t_en=0  Normal:    loop k → weight_in[col] = W[k][col]
//                     → PE[k][col].W_reg = W[k][col]
//
//  t_en=1  Transpose: loop k → weight_in[row] = W[row][k]
//                     → PE[row][k].W_reg = W[row][k]
//
//  Both produce the same final state: PE[row][col] = W[row][col].
// ================================================================
task automatic run_matmul(
    input  logic [DATA_W-1:0]     A     [N][N],
    input  logic [DATA_W-1:0]     W     [N][N],
    input  logic                  t_en,
    output logic [DATA_W_OUT-1:0] C_got [N][N]
);
    transpose_en = t_en;

    // ── START PULSE: IDLE → LOAD_W ──────────────────────────
    // valid_in=1 in IDLE → CU moves to LOAD_W next cycle.
    // load_w=0 this cycle (CU still in IDLE). Nothing loaded.
    for (int c=0; c<N; c++) weight_in[c] = '0;
    valid_in = 1;
    tick;                   // IDLE → LOAD_W

    // ── LOAD_W: N ticks ─────────────────────────────────────
    for (int k=0; k<N; k++) begin
        if (!t_en)
            for (int c=0; c<N; c++) weight_in[c] = W[k][c];   // row k
        else
            for (int r=0; r<N; r++) weight_in[r] = W[r][k];   // col k
        valid_in = 1;
        tick;
    end

    // ── FEED_A: N ticks ─────────────────────────────────────
    for (int c=0; c<N; c++) weight_in[c] = '0;
    for (int i=0; i<N; i++) begin
        for (int r=0; r<N; r++) act_in[r] = A[i][r];
        valid_in = 1;
        tick;
    end

    // ── Wait → capture ──────────────────────────────────────
    valid_in = 0;
    for (int r=0; r<N; r++) act_in[r] = '0;
    while (!valid_out) tick;
    for (int i=0; i<N; i++) begin
        for (int c=0; c<N; c++) C_got[i][c] = psum_out[c];
        tick;
    end
    for (int c=0; c<N; c++) weight_in[c] = '0;
endtask

// ── Golden model ──────────────────────────────────────────────
function automatic void matmul_golden(
    input  logic [DATA_W-1:0]     A [N][N],
    input  logic [DATA_W-1:0]     W [N][N],
    output logic [DATA_W_OUT-1:0] C [N][N]
);
    for (int i=0; i<N; i++)
        for (int col=0; col<N; col++) begin
            C[i][col] = '0;
            for (int row=0; row<N; row++)
                C[i][col] += DATA_W_OUT'(A[i][row]) * DATA_W_OUT'(W[row][col]);
        end
endfunction

// ── Check ─────────────────────────────────────────────────────
task automatic check_matrix(
    input string                 tname,
    input logic [DATA_W_OUT-1:0] C_exp [N][N],
    input logic [DATA_W_OUT-1:0] C_got [N][N]
);
    logic ok; ok=1;
    for (int i=0; i<N; i++)
        for (int col=0; col<N; col++)
            if (C_got[i][col] !== C_exp[i][col]) begin
                $display("    row%0d col%0d: exp=%0d got=%0d",
                         i, col, C_exp[i][col], C_got[i][col]);
                ok = 0;
            end
    if (ok) begin $display("  PASS [%s]", tname); pass_count++; end
    else    begin $display("  FAIL [%s]", tname); fail_count++; end
endtask

// ================================================================
//  TEST CASES
// ================================================================
initial begin
    $dumpfile("SA_NxN_top_tb.vcd");
    $dumpvars(0, SA_NxN_top_tb);

    // ============================================================
    //  TC1  Normal | A=ones | W=ones
    //  C[i][col] = SUM_row(1*1) = 16  for every element.
    //  Basic sanity: pipeline runs end-to-end, valid_out fires.
    // ============================================================
    $display("\n=== TC1: Normal | A=ones | W=ones ===");
    begin
        logic [DATA_W-1:0]     A[N][N], W[N][N];
        logic [DATA_W_OUT-1:0] C_exp[N][N], C_got[N][N];

        for (int i=0;i<N;i++) for (int k=0;k<N;k++) A[i][k] = 8'd1;
        for (int r=0;r<N;r++) for (int c=0;c<N;c++) W[r][c] = 8'd1;

        matmul_golden(A, W, C_exp);
        full_reset();
        run_matmul(A, W, 0, C_got);
        check_matrix("TC1_normal_ones", C_exp, C_got);
    end

    // ============================================================
    //  TC2  Normal | A=3*Identity | W=(r+1)*(c+1)
    //
    //  A[i][k] = 3 if k==i, else 0.
    //  C[i][col] = 3 * W[i][col] = 3*(i+1)*(col+1).
    //
    //  Every output row depends on exactly ONE PE row.
    //  Wrong weight row assignment → wrong C row.
    //  Wrong TRSRL delay → wrong C row.
    // ============================================================
    $display("\n=== TC2: Normal | A=3*Identity | W=(r+1)*(c+1) ===");
    begin
        logic [DATA_W-1:0]     A[N][N], W[N][N];
        logic [DATA_W_OUT-1:0] C_exp[N][N], C_got[N][N];

        for (int i=0;i<N;i++) for (int k=0;k<N;k++) A[i][k] = 8'd0;
        for (int i=0;i<N;i++) A[i][i] = 8'd3;
        for (int r=0;r<N;r++) for (int c=0;c<N;c++) W[r][c] = 8'((r+1)*(c+1));

        matmul_golden(A, W, C_exp);
        full_reset();
        run_matmul(A, W, 0, C_got);
        check_matrix("TC2_normal_identity_A", C_exp, C_got);
    end

    // ============================================================
    //  TC3  Normal | A=zeros | W=unique
    //
    //  A=0 → C must be all-zero regardless of W.
    //  Catches psum not cleared, or load_w triggering accumulation.
    // ============================================================
    $display("\n=== TC3: Normal | A=zeros | W=unique ===");
    begin
        logic [DATA_W-1:0]     A[N][N], W[N][N];
        logic [DATA_W_OUT-1:0] C_exp[N][N], C_got[N][N];

        for (int i=0;i<N;i++) for (int k=0;k<N;k++) A[i][k] = 8'd0;
        for (int r=0;r<N;r++) for (int c=0;c<N;c++) W[r][c] = 8'((r+1)*(c+1));

        matmul_golden(A, W, C_exp);
        full_reset();
        run_matmul(A, W, 0, C_got);
        check_matrix("TC3_normal_zero_A", C_exp, C_got);
    end

    // ============================================================
    //  TC4  Normal | A=unique | W=unique (prime modulus)
    //
    //  W[r][c] = (7*r + 3*c + 1) % 251
    //  A[i][k] = (5*i + 11*k + 2) % 251
    //
    //  No repeated values. No algebraic structure.
    //  Any wrong weight in any single PE → wrong output element.
    // ============================================================
    $display("\n=== TC4: Normal | A=prime | W=prime ===");
    begin
        logic [DATA_W-1:0]     A[N][N], W[N][N];
        logic [DATA_W_OUT-1:0] C_exp[N][N], C_got[N][N];

        for (int i=0;i<N;i++) for (int k=0;k<N;k++) A[i][k] = 8'((5*i + 11*k + 2) % 251);
        for (int r=0;r<N;r++) for (int c=0;c<N;c++) W[r][c] = 8'((7*r +  3*c + 1) % 251);

        matmul_golden(A, W, C_exp);
        full_reset();
        run_matmul(A, W, 0, C_got);
        check_matrix("TC4_normal_prime", C_exp, C_got);
    end

    // ============================================================
    //  TC5  Transpose | A=3*Identity | W=(r+1)*(c+1)
    //
    //  Same W and A as TC2, but W loaded via transpose path.
    //
    //  Transpose feed (ascending col order):
    //    k=0:   weight_in[row] = W[row][0]   → PE[row][0]  (LEFT)
    //    k=1:   weight_in[row] = W[row][1]   → PE[row][1]
    //    ...
    //    k=N-1: weight_in[row] = W[row][N-1] → PE[row][N-1] (RIGHT)
    //
    //  After loading: PE[row][col].W_reg = W[row][col]  ← same as TC2.
    //  Expected result = TC2: C[i][col] = 3*(i+1)*(col+1).
    //
    //  If result differs from TC2:
    //    → weight_L_sig direction wrong
    //    → w_out_right mux in PE broken
    //    → column index assignment wrong
    // ============================================================
    $display("\n=== TC5: Transpose | A=3*Identity | W=(r+1)*(c+1) ===");
    $display("  Same W as TC2 loaded via transpose path");
    $display("  Expected: C[i][col] = 3*(i+1)*(col+1)  (identical to TC2)");
    begin
        logic [DATA_W-1:0]     A[N][N], W[N][N];
        logic [DATA_W_OUT-1:0] C_exp[N][N], C_got[N][N];

        for (int i=0;i<N;i++) for (int k=0;k<N;k++) A[i][k] = 8'd0;
        for (int i=0;i<N;i++) A[i][i] = 8'd3;
        for (int r=0;r<N;r++) for (int c=0;c<N;c++) W[r][c] = 8'((r+1)*(c+1));

        matmul_golden(A, W, C_exp);
        full_reset();
        run_matmul(A, W, 1, C_got);   // t_en=1 → transpose
        check_matrix("TC5_transpose_match_TC2", C_exp, C_got);
    end

    //  TC6  Transpose | A=unique | W=unique
    $display("\n=== TC6: Transpose | A=unique | W=unique");
    $display("  loaded via transpose path");
    begin
        logic [DATA_W-1:0]     A[N][N], W[N][N];
        logic [DATA_W_OUT-1:0] C_exp[N][N], C_got[N][N];

        for (int i=0;i<N;i++) for (int k=0;k<N;k++) A[i][k] = 8'((i+1)*(k+1));
        for (int r=0;r<N;r++) for (int c=0;c<N;c++) W[r][c] = 8'((r+1)*(c+1));
        matmul_golden(A, W, C_exp);
        full_reset();
        run_matmul(A, W, 1, C_got);   // t_en=1 → transpose
        check_matrix("TC6_transpose_unique", C_exp, C_got);
    end

    // ================================================================
    //  RANDOM — 1000 fully randomized matmul tests
    //  Each test: full_reset + random A + random W + golden check
    // ================================================================
    $display("\n=== RAND: 1000 randomized matmul tests ===");
    begin
        logic [DATA_W-1:0]     A[N][N], W[N][N];
        logic [DATA_W_OUT-1:0] C_exp[N][N], C_got[N][N];
        int rand_pass = 0;
        int rand_fail = 0;
        logic t_en;
        for (int t = 0; t < 1000; t++) begin
            // Randomize
            for (int i=0;i<N;i++)
                for (int k=0;k<N;k++)
                    A[i][k] = $urandom_range(0, 255);
            for (int r=0;r<N;r++)
                for (int c=0;c<N;c++)
                    W[r][c] = $urandom_range(0, 255);

            t_en = $urandom_range(0,1);

            // Golden reference
            matmul_golden(A, W, C_exp);

            // DUT run 
            NO_reset();
            run_matmul(A, W, t_en, C_got);

            // Check 
            begin
                logic ok; ok = 1;
                for (int i=0;i<N;i++)
                    for (int col=0;col<N;col++)
                        if (C_got[i][col] !== C_exp[i][col]) ok = 0;

                if (ok) begin
                    rand_pass++;
                    pass_count++;
                    if ((t+1) % 100 == 0)
                        $display("  [RAND] %0d/1000 done — all passed so far", t+1);
                end else begin
                    rand_fail++;
                    fail_count++;
                    $display("  FAIL [RAND_t%0d]", t);
                    // Show first 3 mismatches to keep log readable
                    begin
                        int shown; shown = 0;
                        for (int i=0;i<N;i++)
                            for (int col=0;col<N;col++)
                                if (C_got[i][col] !== C_exp[i][col] && shown < 3) begin
                                    $display("    row%0d col%0d: exp=%0d got=%0d",
                                             i, col, C_exp[i][col], C_got[i][col]);
                                    shown++;
                                end
                    end
                end
            end

        end

        $display("\n  RAND result: %0d PASS  %0d FAIL  (out of 1000)",
                 rand_pass, rand_fail);
    end

    // ── Summary ───────────────────────────────────────────────
    $display("\n================================================");
    $display("  TOTAL: %0d PASS   %0d FAIL", pass_count, fail_count);
    $display("================================================");
    if (fail_count == 0) $display("  >>> ALL TESTS PASSED <<<");
    else                 $display("  >>> FAILURES DETECTED <<<");

    $finish;
end

// ── Timeout ───────────────────────────────────────────────────
initial begin
    #(6*1000 * (4*N + 10) * 20 * 10);
    $display("TIMEOUT"); $finish;
end

endmodule