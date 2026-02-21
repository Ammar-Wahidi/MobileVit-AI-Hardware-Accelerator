`timescale 1ns/1ps

// ================================================================
//  Lego_SA_tb — System testbench for Lego_SA  (10 test cases)
//
// ================================================================
//  ── TRANSPOSE MODE EXPLAINED ──────────────────────────────────
//
//  Both normal and transpose weight loading produce IDENTICAL
//  PE storage and therefore IDENTICAL output: C = A @ W.
//  Transpose mode changes the PHYSICAL WIRING PATH, not the math.
//
//  Normal    (transpose_en=0): weights enter from the BOTTOM edge
//    and shift UP one row per clock.
//    Load tick k → weight_in[col] = W[k][col]  (drive ROW k)
//    After N ticks: PE[row][col] = W[row][col]  ✓
//
//  Transpose (transpose_en=1): weights enter from the RIGHT edge
//    and shift LEFT one column per clock.
//    Load tick k → weight_in[row] = W[row][k]  (drive COLUMN k)
//    After N ticks: PE[row][col] = W[row][col]  ✓
//
//  Both paths produce the same PE state → C = A @ W always.
//  This matches SA_NxN_top_tb TC5 which confirms:
//    "After loading: PE[row][col].W_reg = W[row][col] — same as TC2."
//
// ================================================================
//  ── TRANSPOSE COLUMN PACKING (weight_in bus) ──────────────────
//
//  weight_in[0:N-1]   → RU tile weight input
//  weight_in[N:2N-1]  → LU tile weight input
//  weight_in[2N:3N-1] → RD tile weight input
//  weight_in[3N:4N-1] → LD tile weight input
//
//  TYPE 0  W(16×64): 4 tiles, each gets a 16-col block of W.
//    Transpose tick k: drive column k of each tile block:
//      weight_in[r]    = W[r][k]       (RU tile: column k, row r)
//      weight_in[N+r]  = W[r][N+k]     (LU tile: column k, row r)
//      weight_in[2N+r] = W[r][2N+k]    (RD tile: column k, row r)
//      weight_in[3N+r] = W[r][3N+k]    (LD tile: column k, row r)
//
//  TYPE 1  W(32×32): tiles are RU=W[0:N,0:N], LU=W[0:N,N:2N],
//                              RD=W[N:2N,0:N], LD=W[N:2N,N:2N]
//    Transpose tick k: drive column k of each tile block:
//      weight_in[r]    = W[r][k]       (RU tile col k, top block row r)
//      weight_in[N+r]  = W[r][N+k]     (LU tile col k, top block row r)
//      weight_in[2N+r] = W[N+r][k]     (RD tile col k, bot block row r)
//      weight_in[3N+r] = W[N+r][N+k]   (LD tile col k, bot block row r)
//
//  TYPE 2  W(64×16): tiles are RU=W[0:N], LU=W[2N:3N],
//                              RD=W[N:2N], LD=W[3N:4N]
//    Bus routing: weight_in[0:N-1]→RU, [N:2N-1]→LU, [2N:3N-1]→RD, [3N:4N-1]→LD
//    Normal tick k: drive row k of each tile's weight block:
//      weight_in[c]    = W[c      ][:]  c=0..N-1    (RU, rows  0..N-1)
//      weight_in[N+c]  = W[2N+c   ][:]  c=0..N-1    (LU, rows 2N..3N-1)
//      weight_in[2N+c] = W[N+c    ][:]  c=0..N-1    (RD, rows  N..2N-1)
//      weight_in[3N+c] = W[3N+c   ][:]  c=0..N-1    (LD, rows 3N..4N-1)
//    Transpose tick k: drive column k of each tile block:
//      weight_in[r]    = W[r][k]       (RU tile col k)
//      weight_in[N+r]  = W[2N+r][k]   (LU tile col k)
//      weight_in[2N+r] = W[N+r][k]    (RD tile col k)
//      weight_in[3N+r] = W[3N+r][k]   (LD tile col k)
//
// ================================================================
//  ── ACTIVATION BUS PACKING ────────────────────────────────────
//
//  TYPE 0: act_in[0:N-1] = A[i][0:N-1]  →  broadcast to all tiles
//  TYPE 1: act_in[0:N-1]  = A[i][0:N-1]   → RU,LU
//          act_in[N:2N-1] = A[i][N:2N-1]  → RD,LD
//  TYPE 2: act_in[0:N-1]   = A[i][0:N-1]    → RU
//          act_in[N:2N-1]  = A[i][N:2N-1]   → RD
//          act_in[2N:3N-1] = A[i][2N:3N-1]  → LU
//          act_in[3N:4N-1] = A[i][3N:4N-1]  → LD
//  (activation packing is the same for normal and transpose)
//
// ================================================================
//  ── PROTOCOL (single-CU design) ───────────────────────────────
//
//  1. START TICK:  valid_in=1, zero data → IDLE→LOAD_W (load_w=0 this tick)
//  2. LOAD_W (N ticks): valid_in=1, drive weight_in each tick
//  3. FEED_A (N ticks): valid_in=1, drive act_in each tick
//  4. DRAIN (N-1 cycles, autonomous): valid_in=0, wait for valid_out
//  5. OUTPUT (N cycles): read psum_out then tick for each row
//  6. Wait for busy=0 before next test
//
// ================================================================
//  ── TEST PLAN ─────────────────────────────────────────────────
//
//  TC1   TYPE0 normal    A=ones,     W=ones       → C[i][j]=16
//  TC2   TYPE0 normal    A=3·I,      W[r][c]=r+1  → C[i][j]=3(i+1)
//  TC3   TYPE1 normal    A=ones,     W=ones       → C[i][j]=32
//  TC4   TYPE1 normal    A=diag,     W=ones       → C[i][j]=i+1
//  TC5   TYPE2 normal    A=ones,     W=ones       → C[i][j]=64
//  TC6   TYPE0 TRANSPOSE A=3·I,      W[r][c]=r+1  → C[i][j]=3(i+1)
//           Same matrices as TC2, column-path load → same result.
//           Verifies: w_in_left / w_out_right wiring, weight_L_sig routing.
//  TC7   TYPE0 normal    A=unique,   W=unique     → C=A@W (software golden)
//           No algebraic symmetry: any wrong PE weight → unique wrong output.
//  TC8   TYPE1 TRANSPOSE A=unique,   W=unique     → C=A@W
//           TYPE1 tile layout + transpose column load; 4 tile blocks each
//           loaded column-by-column via left-shift path.
//  TC9   TYPE2 normal    A=unique,   W=unique     → C=A@W
//           All 4×16×16 = 1024 PEs exercised with unique values.
//  TC10  TYPE2 TRANSPOSE A=unique,   W=unique     → C=A@W (same as TC9)
//           Combines TYPE2 4-way layout with transpose path; must match TC9.
//
// ================================================================

module Lego_SA_tb;

// ── Parameters ────────────────────────────────────────────────────
localparam DATA_W     = 8;
localparam DATA_W_OUT = 32;
localparam N          = 16;
localparam TOTAL      = 4 * N;   // 64

// ── DUT signals ───────────────────────────────────────────────────
logic                  clk;
logic                  rst_n;
logic                  valid_in;
logic [1:0]            lego_type;
logic [7:0]            y_input_size;
logic                  transpose_en;
logic [DATA_W-1:0]     act_in    [TOTAL];
logic [DATA_W-1:0]     weight_in [TOTAL];
logic [DATA_W_OUT-1:0] psum_out  [TOTAL];
logic                  valid_out;
logic                  busy;

// ── DUT ───────────────────────────────────────────────────────────
Lego_SA #(
    .DATA_W    (DATA_W    ),
    .DATA_W_OUT(DATA_W_OUT),
    .N_TILE    (N         )
) dut (
    .clk         (clk         ),
    .rst_n       (rst_n       ),
    .valid_in    (valid_in    ),
    .lego_type   (lego_type   ),
    .y_input_size(y_input_size),
    .transpose_en(transpose_en),
    .act_in      (act_in      ),
    .weight_in   (weight_in   ),
    .psum_out    (psum_out    ),
    .valid_out   (valid_out   ),
    .busy        (busy        )
);

// ── Clock: 10 ns period ───────────────────────────────────────────
initial clk = 0;
always  #5 clk = ~clk;

// ── Scoreboard ────────────────────────────────────────────────────
int pass_cnt = 0;
int fail_cnt = 0;

// ── Random sweep variables (module-level — ModelSim requires all   ──
//    declarations before any procedural statements in a begin block) ──
int          rand_t;
int          rand_mode;
int          rand_i;
int          rand_j;
int unsigned rand_seed;
int unsigned rand_seed_tmp;
string       rand_label;

logic [DATA_W-1:0]     rA0  [N][N];
logic [DATA_W-1:0]     rW0  [N][TOTAL];
logic [DATA_W_OUT-1:0] rC0  [N][TOTAL];
logic [DATA_W_OUT-1:0] rE0  [N][TOTAL];

logic [DATA_W-1:0]     rA1  [N][2*N];
logic [DATA_W-1:0]     rW1  [2*N][2*N];
logic [DATA_W_OUT-1:0] rC1  [N][2*N];
logic [DATA_W_OUT-1:0] rE1  [N][2*N];

logic [DATA_W-1:0]     rA2  [N][TOTAL];
logic [DATA_W-1:0]     rW2  [TOTAL][N];
logic [DATA_W_OUT-1:0] rC2  [N][N];
logic [DATA_W_OUT-1:0] rE2  [N][N];

logic [DATA_W_OUT-1:0] rand_exp [TOTAL*TOTAL];
logic [DATA_W_OUT-1:0] rand_got [TOTAL*TOTAL];

// ─────────────────────────────────────────────────────────────────
// tick — advance one posedge, then settle 1 ns
// ─────────────────────────────────────────────────────────────────
task automatic tick;
    @(posedge clk); #1;
endtask

// ─────────────────────────────────────────────────────────────────
// zero_buses
// ─────────────────────────────────────────────────────────────────
task automatic zero_buses;
    for (int i = 0; i < TOTAL; i++) begin
        weight_in[i] = '0;
        act_in[i]    = '0;
    end
endtask

// ─────────────────────────────────────────────────────────────────
// reset_dut
//   rst_n=0 for 4 cycles, then release and wait for busy=0.
//   valid_in held 0 throughout so FSM never re-triggers.
// ─────────────────────────────────────────────────────────────────
task automatic reset_dut;
    rst_n        = 0;
    valid_in     = 0;
    transpose_en = 0;
    lego_type    = '0;
    y_input_size = 8'(N);
    zero_buses();
    repeat(4) tick;
    rst_n = 1;
    while (busy) tick;
    tick;
endtask

// ─────────────────────────────────────────────────────────────────
// do_start_tick
//   One tick with valid_in=1 and zero data: IDLE → LOAD_W.
//   load_w stays 0 this tick (state is still IDLE at posedge).
//   After: state=LOAD_W, cnt=0, load_w=1. Next N ticks load weights.
// ─────────────────────────────────────────────────────────────────
task automatic do_start_tick;
    zero_buses();
    valid_in = 1;
    tick;
endtask

// ─────────────────────────────────────────────────────────────────
// do_drain_wait
//   Called after the last FEED_A tick. FSM is already in DRAIN.
//   Just deassert valid_in and wait for valid_out.
// ─────────────────────────────────────────────────────────────────
task automatic do_drain_wait;
    valid_in = 0;
    zero_buses();
    while (!valid_out) tick;
    // On exit: valid_out=1, psum_out holds row 0 of result.
endtask

// ─────────────────────────────────────────────────────────────────
// Capture helpers — read-before-tick for N rows
// ─────────────────────────────────────────────────────────────────
task automatic capture_t0 (output logic [DATA_W_OUT-1:0] C [N][TOTAL]);
    for (int i = 0; i < N; i++) begin
        for (int c = 0; c < TOTAL; c++) C[i][c] = psum_out[c];
        tick;
    end
endtask

task automatic capture_t1 (output logic [DATA_W_OUT-1:0] C [N][2*N]);
    for (int i = 0; i < N; i++) begin
        for (int c = 0; c < 2*N; c++) C[i][c] = psum_out[c];
        tick;
    end
endtask

task automatic capture_t2 (output logic [DATA_W_OUT-1:0] C [N][N]);
    for (int i = 0; i < N; i++) begin
        for (int c = 0; c < N; c++) C[i][c] = psum_out[c];
        tick;
    end
endtask

// ================================================================
//  Software golden reference — pure SV matmul
// ================================================================
function automatic void golden_t0(
    input  logic [DATA_W-1:0]     A [N][N],
    input  logic [DATA_W-1:0]     W [N][TOTAL],
    output logic [DATA_W_OUT-1:0] C [N][TOTAL]
);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < TOTAL; j++) begin
            C[i][j] = '0;
            for (int k = 0; k < N; k++)
                C[i][j] += DATA_W_OUT'(A[i][k]) * DATA_W_OUT'(W[k][j]);
        end
endfunction

function automatic void golden_t1(
    input  logic [DATA_W-1:0]     A [N][2*N],
    input  logic [DATA_W-1:0]     W [2*N][2*N],
    output logic [DATA_W_OUT-1:0] C [N][2*N]
);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < 2*N; j++) begin
            C[i][j] = '0;
            for (int k = 0; k < 2*N; k++)
                C[i][j] += DATA_W_OUT'(A[i][k]) * DATA_W_OUT'(W[k][j]);
        end
endfunction

function automatic void golden_t2(
    input  logic [DATA_W-1:0]     A [N][TOTAL],
    input  logic [DATA_W-1:0]     W [TOTAL][N],
    output logic [DATA_W_OUT-1:0] C [N][N]
);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) begin
            C[i][j] = '0;
            for (int k = 0; k < TOTAL; k++)
                C[i][j] += DATA_W_OUT'(A[i][k]) * DATA_W_OUT'(W[k][j]);
        end
endfunction

// ================================================================
//  run_type0  — TYPE 0 NORMAL
//  A(N×N) × W(N×TOTAL) = C(N×TOTAL)
//
//  Load tick k: drive ROW k of W across all 4 tile slots.
//    weight_in[0:N-1]   = W[k][0:N-1]
//    weight_in[N:2N-1]  = W[k][N:2N-1]
//    weight_in[2N:3N-1] = W[k][2N:3N-1]
//    weight_in[3N:4N-1] = W[k][3N:4N-1]
//  Feed tick i: act_in[0:N-1] = A[i][0:N-1] (broadcast)
// ================================================================
task automatic run_type0 (
    input  logic [DATA_W-1:0]     A     [N][N],
    input  logic [DATA_W-1:0]     W     [N][TOTAL],
    output logic [DATA_W_OUT-1:0] C_got [N][TOTAL]
);
    lego_type    = 2'd0;
    y_input_size = 8'(N);
    transpose_en = 0;

    do_start_tick();

    for (int k = 0; k < N; k++) begin
        for (int c = 0; c < TOTAL; c++) weight_in[c] = W[k][c];
        for (int c = 0; c < TOTAL; c++) act_in[c]    = '0;
        valid_in = 1;
        tick;
    end
    for (int i = 0; i < N; i++) begin
        for (int c = 0;    c < N;     c++) act_in[c] = A[i][c];
        for (int c = N;    c < TOTAL; c++) act_in[c] = '0;
        valid_in = 1;
        tick;
    end
    do_drain_wait();
    capture_t0(C_got);
endtask

// ================================================================
//  run_type0_tr  — TYPE 0 TRANSPOSE
//  Same computation C = A @ W; loads via RIGHT-edge column path.
//
//  Load tick k: drive COLUMN k of each tile's 16-wide block.
//    weight_in[r]    = W[r][k]       r=0..N-1  → RU col k
//    weight_in[N+r]  = W[r][N+k]     r=0..N-1  → LU col k
//    weight_in[2N+r] = W[r][2N+k]    r=0..N-1  → RD col k
//    weight_in[3N+r] = W[r][3N+k]    r=0..N-1  → LD col k
//  Feed tick i: same broadcast as normal
// ================================================================
task automatic run_type0_tr (
    input  logic [DATA_W-1:0]     A     [N][N],
    input  logic [DATA_W-1:0]     W     [N][TOTAL],
    output logic [DATA_W_OUT-1:0] C_got [N][TOTAL]
);
    lego_type    = 2'd0;
    y_input_size = 8'(N);
    transpose_en = 1;

    do_start_tick();

    for (int k = 0; k < N; k++) begin
        for (int r = 0; r < N; r++) begin
            weight_in[r]        = W[r][k];        // RU tile: column k, row r
            weight_in[N+r]      = W[r][N+k];      // LU tile: column k, row r
            weight_in[2*N+r]    = W[r][2*N+k];    // RD tile: column k, row r
            weight_in[3*N+r]    = W[r][3*N+k];    // LD tile: column k, row r
        end
        for (int c = 0; c < TOTAL; c++) act_in[c] = '0;
        valid_in = 1;
        tick;
    end
    for (int i = 0; i < N; i++) begin
        for (int c = 0;    c < N;     c++) act_in[c] = A[i][c];
        for (int c = N;    c < TOTAL; c++) act_in[c] = '0;
        valid_in = 1;
        tick;
    end
    do_drain_wait();
    capture_t0(C_got);
    transpose_en = 0;
endtask

// ================================================================
//  run_type1  — TYPE 1 NORMAL
//  A(N×2N) × W(2N×2N) = C(N×2N)
//  Tiles: RU=W[0:N,0:N], LU=W[0:N,N:2N], RD=W[N:2N,0:N], LD=W[N:2N,N:2N]
//
//  Load tick k (interleaved rows k and k+N):
//    weight_in[0:N-1]   = W[k][0:N-1]       → RU
//    weight_in[N:2N-1]  = W[k][N:2N-1]      → LU
//    weight_in[2N:3N-1] = W[k+N][0:N-1]     → RD
//    weight_in[3N:4N-1] = W[k+N][N:2N-1]    → LD
//  Feed tick i:
//    act_in[0:N-1]  = A[i][0:N-1]   → RU, LU
//    act_in[N:2N-1] = A[i][N:2N-1]  → RD, LD
//  Output: psum_out[0:N-1]=RU+RD, psum_out[N:2N-1]=LU+LD
// ================================================================
task automatic run_type1 (
    input  logic [DATA_W-1:0]     A     [N][2*N],
    input  logic [DATA_W-1:0]     W     [2*N][2*N],
    output logic [DATA_W_OUT-1:0] C_got [N][2*N]
);
    lego_type    = 2'd1;
    y_input_size = 8'(N);
    transpose_en = 0;

    do_start_tick();

    for (int k = 0; k < N; k++) begin
        for (int c = 0;   c < N;    c++) weight_in[c]    = W[k  ][c      ];   // RU
        for (int c = N;   c < 2*N;  c++) weight_in[c]    = W[k  ][c      ];   // LU
        for (int c = 2*N; c < 3*N;  c++) weight_in[c]    = W[k+N][c-2*N  ];   // RD
        for (int c = 3*N; c < TOTAL;c++) weight_in[c]    = W[k+N][N+(c-3*N)]; // LD
        for (int c = 0;   c < TOTAL;c++) act_in[c]       = '0;
        valid_in = 1;
        tick;
    end
    for (int i = 0; i < N; i++) begin
        for (int c = 0;   c < 2*N;  c++) act_in[c] = A[i][c];
        for (int c = 2*N; c < TOTAL;c++) act_in[c] = '0;
        valid_in = 1;
        tick;
    end
    do_drain_wait();
    capture_t1(C_got);
endtask

// ================================================================
//  run_type1_tr  — TYPE 1 TRANSPOSE
//  Same computation C = A @ W; loads via RIGHT-edge column path.
//
//  Load tick k: drive COLUMN k of each tile's 16×16 block.
//    weight_in[r]    = W[r][k]       r=0..N-1  → RU tile col k (top-left)
//    weight_in[N+r]  = W[r][N+k]     r=0..N-1  → LU tile col k (top-right)
//    weight_in[2N+r] = W[N+r][k]     r=0..N-1  → RD tile col k (bot-left)
//    weight_in[3N+r] = W[N+r][N+k]   r=0..N-1  → LD tile col k (bot-right)
// ================================================================
task automatic run_type1_tr (
    input  logic [DATA_W-1:0]     A     [N][2*N],
    input  logic [DATA_W-1:0]     W     [2*N][2*N],
    output logic [DATA_W_OUT-1:0] C_got [N][2*N]
);
    lego_type    = 2'd1;
    y_input_size = 8'(N);
    transpose_en = 1;

    do_start_tick();

    for (int k = 0; k < N; k++) begin
        for (int r = 0; r < N; r++) begin
            weight_in[r]        = W[r][k];        // RU tile col k
            weight_in[N+r]      = W[r][N+k];      // LU tile col k
            weight_in[2*N+r]    = W[N+r][k];      // RD tile col k
            weight_in[3*N+r]    = W[N+r][N+k];    // LD tile col k
        end
        for (int c = 0; c < TOTAL; c++) act_in[c] = '0;
        valid_in = 1;
        tick;
    end
    for (int i = 0; i < N; i++) begin
        for (int c = 0;   c < 2*N;  c++) act_in[c] = A[i][c];
        for (int c = 2*N; c < TOTAL;c++) act_in[c] = '0;
        valid_in = 1;
        tick;
    end
    do_drain_wait();
    capture_t1(C_got);
    transpose_en = 0;
endtask

// ================================================================
//  run_type2  — TYPE 2 NORMAL
//  A(N×TOTAL) × W(TOTAL×N) = C(N×N)
//
//  Hardware routing (Lego_SA.sv, always_comb weight block):
//    tile_w_RU = weight_in[0:N-1]
//    tile_w_LU = weight_in[N:2N-1]      ← slot N goes to LU
//    tile_w_RD = weight_in[2N:3N-1]     ← slot 2N goes to RD
//    tile_w_LD = weight_in[3N:4N-1]
//
//  Tile–weight assignment for C = A @ W (golden_t2):
//    RU acts = A[i][0:N-1]    → needs W rows   0..N-1   → bus slot 0
//    LU acts = A[i][2N:3N-1]  → needs W rows  2N..3N-1  → bus slot N  (LU!)
//    RD acts = A[i][N:2N-1]   → needs W rows   N..2N-1  → bus slot 2N (RD!)
//    LD acts = A[i][3N:4N-1]  → needs W rows  3N..4N-1  → bus slot 3N
//
//  Load tick k:
//    weight_in[0:N-1]   = W[k     ][:] → RU gets rows   0..N-1  ✓
//    weight_in[N:2N-1]  = W[k+2N  ][:] → LU gets rows  2N..3N-1 ✓
//    weight_in[2N:3N-1] = W[k+N   ][:] → RD gets rows   N..2N-1 ✓
//    weight_in[3N:4N-1] = W[k+3N  ][:] → LD gets rows  3N..4N-1 ✓
// ================================================================
task automatic run_type2 (
    input  logic [DATA_W-1:0]     A     [N][TOTAL],
    input  logic [DATA_W-1:0]     W     [TOTAL][N],
    output logic [DATA_W_OUT-1:0] C_got [N][N]
);
    lego_type    = 2'd2;
    y_input_size = 8'(N);
    transpose_en = 0;

    do_start_tick();

    for (int k = 0; k < N; k++) begin
        for (int c = 0;   c < N;    c++) weight_in[c]    = W[k      ][c    ]; // RU
        for (int c = N;   c < 2*N;  c++) weight_in[c]    = W[k + 2*N][c-  N]; // LU  (slot N → LU tile)
        for (int c = 2*N; c < 3*N;  c++) weight_in[c]    = W[k +  N ][c-2*N]; // RD  (slot 2N → RD tile)
        for (int c = 3*N; c < TOTAL;c++) weight_in[c]    = W[k + 3*N][c-3*N]; // LD
        for (int c = 0;   c < TOTAL;c++) act_in[c]       = '0;
        valid_in = 1;
        tick;
    end
    for (int i = 0; i < N; i++) begin
        for (int c = 0; c < TOTAL; c++) act_in[c] = A[i][c];
        valid_in = 1;
        tick;
    end
    do_drain_wait();
    capture_t2(C_got);
endtask

// ================================================================
//  run_type2_tr  — TYPE 2 TRANSPOSE
//  Same computation C = A @ W; loads via RIGHT-edge column path.
//
//  Same hardware routing applies: weight_in[N:2N-1] → LU tile,
//  weight_in[2N:3N-1] → RD tile.
//
//  Load tick k: drive COLUMN k of each tile's 16×16 weight block.
//    weight_in[r]    = W[r][k]       r=0..N-1  → RU tile col k (rows   0..N-1)
//    weight_in[N+r]  = W[2N+r][k]   r=0..N-1  → LU tile col k (rows 2N..3N-1) ✓
//    weight_in[2N+r] = W[N+r][k]    r=0..N-1  → RD tile col k (rows  N..2N-1) ✓
//    weight_in[3N+r] = W[3N+r][k]   r=0..N-1  → LD tile col k (rows 3N..4N-1)
// ================================================================
task automatic run_type2_tr (
    input  logic [DATA_W-1:0]     A     [N][TOTAL],
    input  logic [DATA_W-1:0]     W     [TOTAL][N],
    output logic [DATA_W_OUT-1:0] C_got [N][N]
);
    lego_type    = 2'd2;
    y_input_size = 8'(N);
    transpose_en = 1;

    do_start_tick();

    for (int k = 0; k < N; k++) begin
        for (int r = 0; r < N; r++) begin
            weight_in[r]        = W[r][k];        // RU tile col k (rows   0..N-1)
            weight_in[N+r]      = W[2*N+r][k];    // LU tile col k (rows 2N..3N-1) ✓
            weight_in[2*N+r]    = W[N+r][k];      // RD tile col k (rows  N..2N-1) ✓
            weight_in[3*N+r]    = W[3*N+r][k];    // LD tile col k (rows 3N..4N-1)
        end
        for (int c = 0; c < TOTAL; c++) act_in[c] = '0;
        valid_in = 1;
        tick;
    end
    for (int i = 0; i < N; i++) begin
        for (int c = 0; c < TOTAL; c++) act_in[c] = A[i][c];
        valid_in = 1;
        tick;
    end
    do_drain_wait();
    capture_t2(C_got);
    transpose_en = 0;
endtask

// ================================================================
//  check_result
// ================================================================
task automatic check_result (
    input string                  tc_name,
    input logic [DATA_W_OUT-1:0]  exp [],
    input logic [DATA_W_OUT-1:0]  got [],
    input int                     rows,
    input int                     cols
);
    logic ok;
    ok = 1;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) begin
            automatic int idx = i * cols + j;
            if (got[idx] !== exp[idx]) begin
                if (ok) $display("    MISMATCHES in [%s]:", tc_name);
                $display("      row=%0d col=%0d : exp=%0d  got=%0d",
                          i, j, exp[idx], got[idx]);
                ok = 0;
            end
        end
    if (ok) begin
        $display("  PASS  [%s]", tc_name);
        pass_cnt++;
    end else begin
        $display("  FAIL  [%s]", tc_name);
        fail_cnt++;
    end
endtask

// ================================================================
//  MAIN
// ================================================================
initial begin
    $dumpfile("Lego_SA_tb.vcd");
    $dumpvars(0, Lego_SA_tb);

    // ─────────────────────────────────────────────────────────────
    //  TC1  TYPE0 normal | A=ones(16×16) | W=ones(16×64)
    //  C[i][j] = 16 everywhere.
    //  Checks: basic end-to-end; valid_out fires N cycles; all 4
    //          tile column outputs concatenated correctly in psum_out.
    // ─────────────────────────────────────────────────────────────
    $display("\n=== TC1: TYPE0 normal | A=ones | W=ones  →  C[i][j]=16 ===");
    begin
        logic [DATA_W-1:0]     A    [N][N];
        logic [DATA_W-1:0]     W    [N][TOTAL];
        logic [DATA_W_OUT-1:0] C_got[N][TOTAL];
        logic [DATA_W_OUT-1:0] C_exp[N][TOTAL];
        logic [DATA_W_OUT-1:0] exp_flat[N*TOTAL];
        logic [DATA_W_OUT-1:0] got_flat[N*TOTAL];

        foreach (A[i,k]) A[i][k] = 8'd1;
        foreach (W[r,c]) W[r][c] = 8'd1;
        golden_t0(A, W, C_exp);

        reset_dut();
        run_type0(A, W, C_got);
        while (busy) tick;

        for (int i=0;i<N;i++) for (int j=0;j<TOTAL;j++) begin
            exp_flat[i*TOTAL+j] = C_exp[i][j];
            got_flat[i*TOTAL+j] = C_got[i][j];
        end
        check_result("TC1_type0_normal_ones", exp_flat, got_flat, N, TOTAL);
    end

    // ─────────────────────────────────────────────────────────────
    //  TC2  TYPE0 normal | A=3·I | W[r][c]=r+1
    //  C[i][j] = 3*(i+1) for all j (constant per row).
    //  Checks: W[0] is loaded (row 0 test); correct PE row assignment;
    //          each tile receives the right column slice of W.
    // ─────────────────────────────────────────────────────────────
    $display("\n=== TC2: TYPE0 normal | A=3*I | W[r][c]=r+1  →  C[i][j]=3*(i+1) ===");
    begin
        logic [DATA_W-1:0]     A    [N][N];
        logic [DATA_W-1:0]     W    [N][TOTAL];
        logic [DATA_W_OUT-1:0] C_got[N][TOTAL];
        logic [DATA_W_OUT-1:0] C_exp[N][TOTAL];
        logic [DATA_W_OUT-1:0] exp_flat[N*TOTAL];
        logic [DATA_W_OUT-1:0] got_flat[N*TOTAL];

        foreach (A[i,k]) A[i][k] = (k==i) ? 8'd3 : 8'd0;
        foreach (W[r,c]) W[r][c] = 8'(r+1);
        golden_t0(A, W, C_exp);

        reset_dut();
        run_type0(A, W, C_got);
        while (busy) tick;

        for (int i=0;i<N;i++) for (int j=0;j<TOTAL;j++) begin
            exp_flat[i*TOTAL+j] = C_exp[i][j];
            got_flat[i*TOTAL+j] = C_got[i][j];
        end
        check_result("TC2_type0_normal_3I", exp_flat, got_flat, N, TOTAL);
    end

    // ─────────────────────────────────────────────────────────────
    //  TC3  TYPE1 normal | A=ones(16×32) | W=ones(32×32)
    //  C[i][j] = 32 everywhere.
    //  Checks: TYPE1 pipeline; interleaved load (row k and k+16
    //          at same tick); RU+RD adder = 16+16; LU+LD adder = 16+16.
    // ─────────────────────────────────────────────────────────────
    $display("\n=== TC3: TYPE1 normal | A=ones | W=ones  →  C[i][j]=32 ===");
    begin
        logic [DATA_W-1:0]     A    [N][2*N];
        logic [DATA_W-1:0]     W    [2*N][2*N];
        logic [DATA_W_OUT-1:0] C_got[N][2*N];
        logic [DATA_W_OUT-1:0] C_exp[N][2*N];
        logic [DATA_W_OUT-1:0] exp_flat[N*2*N];
        logic [DATA_W_OUT-1:0] got_flat[N*2*N];

        foreach (A[i,k]) A[i][k] = 8'd1;
        foreach (W[r,c]) W[r][c] = 8'd1;
        golden_t1(A, W, C_exp);

        reset_dut();
        run_type1(A, W, C_got);
        while (busy) tick;

        for (int i=0;i<N;i++) for (int j=0;j<2*N;j++) begin
            exp_flat[i*2*N+j] = C_exp[i][j];
            got_flat[i*2*N+j] = C_got[i][j];
        end
        check_result("TC3_type1_normal_ones", exp_flat, got_flat, N, 2*N);
    end

    // ─────────────────────────────────────────────────────────────
    //  TC4  TYPE1 normal | A=diag(1..16) in 16×32 | W=ones(32×32)
    //  A[i][k] = (i+1) if k==i else 0 (nonzero only top-left N×N).
    //  C[i][j] = i+1 for all j.
    //  Checks: interleaved weight-row ordering. A activates only rows
    //          0..15, so only RU/LU contribute. Wrong row interleaving
    //          → output halved or wrong.
    // ─────────────────────────────────────────────────────────────
    $display("\n=== TC4: TYPE1 normal | A=diag | W=ones  →  C[i][j]=i+1 ===");
    begin
        logic [DATA_W-1:0]     A    [N][2*N];
        logic [DATA_W-1:0]     W    [2*N][2*N];
        logic [DATA_W_OUT-1:0] C_got[N][2*N];
        logic [DATA_W_OUT-1:0] C_exp[N][2*N];
        logic [DATA_W_OUT-1:0] exp_flat[N*2*N];
        logic [DATA_W_OUT-1:0] got_flat[N*2*N];

        foreach (A[i,k]) A[i][k] = (k==i) ? 8'(i+1) : 8'd0;
        foreach (W[r,c]) W[r][c] = 8'd1;
        golden_t1(A, W, C_exp);

        reset_dut();
        run_type1(A, W, C_got);
        while (busy) tick;

        for (int i=0;i<N;i++) for (int j=0;j<2*N;j++) begin
            exp_flat[i*2*N+j] = C_exp[i][j];
            got_flat[i*2*N+j] = C_got[i][j];
        end
        check_result("TC4_type1_normal_diag", exp_flat, got_flat, N, 2*N);
    end

    // ─────────────────────────────────────────────────────────────
    //  TC5  TYPE2 normal | A=ones(16×64) | W=ones(64×16)
    //  C[i][j] = 64 everywhere.
    //  Checks: TYPE2; 4-way interleaved load (rows k/k+16/k+32/k+48
    //          at same tick); 4-way act split; RU+RD+LU+LD adder=64.
    // ─────────────────────────────────────────────────────────────
    $display("\n=== TC5: TYPE2 normal | A=ones | W=ones  →  C[i][j]=64 ===");
    begin
        logic [DATA_W-1:0]     A    [N][TOTAL];
        logic [DATA_W-1:0]     W    [TOTAL][N];
        logic [DATA_W_OUT-1:0] C_got[N][N];
        logic [DATA_W_OUT-1:0] C_exp[N][N];
        logic [DATA_W_OUT-1:0] exp_flat[N*N];
        logic [DATA_W_OUT-1:0] got_flat[N*N];

        foreach (A[i,k]) A[i][k] = 8'd1;
        foreach (W[r,c]) W[r][c] = 8'd1;
        golden_t2(A, W, C_exp);

        reset_dut();
        run_type2(A, W, C_got);
        while (busy) tick;

        for (int i=0;i<N;i++) for (int j=0;j<N;j++) begin
            exp_flat[i*N+j] = C_exp[i][j];
            got_flat[i*N+j] = C_got[i][j];
        end
        check_result("TC5_type2_normal_ones", exp_flat, got_flat, N, N);
    end

    // ─────────────────────────────────────────────────────────────
    //  TC6  TYPE0 TRANSPOSE | A=3·I | W[r][c]=r+1
    //
    //  IDENTICAL matrices to TC2, but weights loaded via COLUMN PATH.
    //  Normal load: drive row k on tick k.
    //  Transpose load: drive column k on tick k (weight_in[r]=W[r][k]).
    //  After N ticks both paths yield PE[row][col]=W[row][col] → same C.
    //
    //  Golden = C = A @ W = same as TC2: C[i][j] = 3*(i+1).
    //
    //  What this catches:
    //    • w_in_left / w_out_right signals in PE.sv wired correctly.
    //    • weight_L_sig routing in SA_NxN.sv correct.
    //    • Column-entry shifts left correctly (right→left each cycle).
    //    • If transpose path broken → output differs from TC2.
    // ─────────────────────────────────────────────────────────────
    $display("\n=== TC6: TYPE0 TRANSPOSE | A=3*I | W[r][c]=r+1  →  C[i][j]=3*(i+1) ===");
    $display("    (same matrices as TC2; result must be identical to TC2)");
    begin
        logic [DATA_W-1:0]     A    [N][N];
        logic [DATA_W-1:0]     W    [N][TOTAL];
        logic [DATA_W_OUT-1:0] C_got[N][TOTAL];
        logic [DATA_W_OUT-1:0] C_exp[N][TOTAL];
        logic [DATA_W_OUT-1:0] exp_flat[N*TOTAL];
        logic [DATA_W_OUT-1:0] got_flat[N*TOTAL];

        foreach (A[i,k]) A[i][k] = (k==i) ? 8'd3 : 8'd0;
        foreach (W[r,c]) W[r][c] = 8'(r+1);
        golden_t0(A, W, C_exp);   // C = A @ W, identical to TC2 golden

        reset_dut();
        run_type0_tr(A, W, C_got);
        while (busy) tick;

        for (int i=0;i<N;i++) for (int j=0;j<TOTAL;j++) begin
            exp_flat[i*TOTAL+j] = C_exp[i][j];
            got_flat[i*TOTAL+j] = C_got[i][j];
        end
        check_result("TC6_type0_transpose_3I", exp_flat, got_flat, N, TOTAL);
    end

    // ─────────────────────────────────────────────────────────────
    //  TC7  TYPE0 normal | A=unique | W=unique (prime-modulus values)
    //
    //  A[i][k] = (5*i + 11*k + 2) % 251   (16×16, max=242)
    //  W[r][c] = (7*r +  3*c + 1) % 251   (16×64, max=250)
    //  Overflow check: 16 × 242 × 250 = 967,200 < 2^32 ✓
    //
    //  Golden = software matmul C = A @ W.
    //
    //  What this catches:
    //    • No repeated values → any wrong PE weight gives unique wrong output.
    //    • All 16 rows of each tile (RU/LU/RD/LD) verified simultaneously.
    //    • Column slice assignment tested across 64-wide output.
    // ─────────────────────────────────────────────────────────────
    $display("\n=== TC7: TYPE0 normal | A=unique | W=unique  →  C=A@W ===");
    begin
        logic [DATA_W-1:0]     A    [N][N];
        logic [DATA_W-1:0]     W    [N][TOTAL];
        logic [DATA_W_OUT-1:0] C_got[N][TOTAL];
        logic [DATA_W_OUT-1:0] C_exp[N][TOTAL];
        logic [DATA_W_OUT-1:0] exp_flat[N*TOTAL];
        logic [DATA_W_OUT-1:0] got_flat[N*TOTAL];

        foreach (A[i,k]) A[i][k] = 8'((5*i + 11*k + 2) % 251);
        foreach (W[r,c]) W[r][c] = 8'((7*r +  3*c + 1) % 251);
        golden_t0(A, W, C_exp);

        reset_dut();
        run_type0(A, W, C_got);
        while (busy) tick;

        for (int i=0;i<N;i++) for (int j=0;j<TOTAL;j++) begin
            exp_flat[i*TOTAL+j] = C_exp[i][j];
            got_flat[i*TOTAL+j] = C_got[i][j];
        end
        check_result("TC7_type0_normal_unique", exp_flat, got_flat, N, TOTAL);
    end

    // ─────────────────────────────────────────────────────────────
    //  TC8  TYPE1 TRANSPOSE | A=unique | W=unique (prime-modulus)
    //
    //  A[i][k] = (5*i + 11*k + 2) % 251   (16×32, max=250)
    //  W[r][c] = (7*r +  3*c + 1) % 251   (32×32, max=250)
    //  Overflow check: 32 × 250 × 250 = 2,000,000 < 2^32 ✓
    //
    //  Golden = software matmul C = A @ W.
    //
    //  What this catches:
    //    • TYPE1 interleaved tile layout (rows k and k+16 simultaneously)
    //      combined with column-order loading.
    //    • All 4 tile blocks (RU/LU/RD/LD) each receive column k of their
    //      16×16 sub-matrix on tick k.
    //    • Wrong column-to-tile mapping → specific output elements wrong.
    // ─────────────────────────────────────────────────────────────
    $display("\n=== TC8: TYPE1 TRANSPOSE | A=unique | W=unique  →  C=A@W ===");
    begin
        logic [DATA_W-1:0]     A    [N][2*N];
        logic [DATA_W-1:0]     W    [2*N][2*N];
        logic [DATA_W_OUT-1:0] C_got[N][2*N];
        logic [DATA_W_OUT-1:0] C_exp[N][2*N];
        logic [DATA_W_OUT-1:0] exp_flat[N*2*N];
        logic [DATA_W_OUT-1:0] got_flat[N*2*N];

        foreach (A[i,k]) A[i][k] = 8'((5*i + 11*k + 2) % 251);
        foreach (W[r,c]) W[r][c] = 8'((7*r +  3*c + 1) % 251);
        golden_t1(A, W, C_exp);

        reset_dut();
        run_type1_tr(A, W, C_got);
        while (busy) tick;

        for (int i=0;i<N;i++) for (int j=0;j<2*N;j++) begin
            exp_flat[i*2*N+j] = C_exp[i][j];
            got_flat[i*2*N+j] = C_got[i][j];
        end
        check_result("TC8_type1_transpose_unique", exp_flat, got_flat, N, 2*N);
    end

    // ─────────────────────────────────────────────────────────────
    //  TC9  TYPE2 normal | A=unique | W=unique (prime-modulus)
    //
    //  A[i][k] = (5*i + 11*k + 2) % 251   (16×64, max=250)
    //  W[r][c] = (7*r +  3*c + 1) % 251   (64×16, max=250)
    //  Overflow check: 64 × 250 × 250 = 4,000,000 < 2^32 ✓
    //
    //  Golden = software matmul C = A @ W.
    //
    //  What this catches:
    //    • All 4 tiles × 16 rows × 16 cols = 1024 PEs, all unique weights.
    //    • 4-way interleaved row loading (rows k, k+16, k+32, k+48 per tick).
    //    • 4-way activation split routing.
    //    • 4-way output adder.
    //    • Any single wrong PE weight → unique wrong output element.
    // ─────────────────────────────────────────────────────────────
    $display("\n=== TC9: TYPE2 normal | A=unique | W=unique  →  C=A@W ===");
    begin
        logic [DATA_W-1:0]     A    [N][TOTAL];
        logic [DATA_W-1:0]     W    [TOTAL][N];
        logic [DATA_W_OUT-1:0] C_got[N][N];
        logic [DATA_W_OUT-1:0] C_exp[N][N];
        logic [DATA_W_OUT-1:0] exp_flat[N*N];
        logic [DATA_W_OUT-1:0] got_flat[N*N];

        foreach (A[i,k]) A[i][k] = 8'((5*i + 11*k + 2) % 251);
        foreach (W[r,c]) W[r][c] = 8'((7*r +  3*c + 1) % 251);
        golden_t2(A, W, C_exp);

        reset_dut();
        run_type2(A, W, C_got);
        while (busy) tick;

        for (int i=0;i<N;i++) for (int j=0;j<N;j++) begin
            exp_flat[i*N+j] = C_exp[i][j];
            got_flat[i*N+j] = C_got[i][j];
        end
        check_result("TC9_type2_normal_unique", exp_flat, got_flat, N, N);
    end

    // ─────────────────────────────────────────────────────────────
    //  TC10  TYPE2 TRANSPOSE | A=unique | W=unique (same as TC9)
    //
    //  Same A and W as TC9; loaded via COLUMN PATH.
    //  Golden = software matmul C = A @ W — must match TC9.
    //
    //  What this catches:
    //    • TYPE2 + transpose: 4 independent 16×16 tiles, each loaded
    //      column-by-column via the left-shift (w_in_left) path.
    //    • Output must exactly match TC9 — any routing error in the
    //      TYPE2 transpose column packing is exposed.
    //    • Specifically validates that the TYPE2 tile-row boundaries
    //      (0/16/32/48) are correctly handled in transpose mode.
    // ─────────────────────────────────────────────────────────────
    $display("\n=== TC10: TYPE2 TRANSPOSE | A=unique | W=unique  →  C=A@W (must match TC9) ===");
    begin
        logic [DATA_W-1:0]     A    [N][TOTAL];
        logic [DATA_W-1:0]     W    [TOTAL][N];
        logic [DATA_W_OUT-1:0] C_got[N][N];
        logic [DATA_W_OUT-1:0] C_exp[N][N];
        logic [DATA_W_OUT-1:0] exp_flat[N*N];
        logic [DATA_W_OUT-1:0] got_flat[N*N];

        foreach (A[i,k]) A[i][k] = 8'((5*i + 11*k + 2) % 251);
        foreach (W[r,c]) W[r][c] = 8'((7*r +  3*c + 1) % 251);
        golden_t2(A, W, C_exp);   // identical golden to TC9

        reset_dut();
        run_type2_tr(A, W, C_got);
        while (busy) tick;

        for (int i=0;i<N;i++) for (int j=0;j<N;j++) begin
            exp_flat[i*N+j] = C_exp[i][j];
            got_flat[i*N+j] = C_got[i][j];
        end
        check_result("TC10_type2_transpose_unique", exp_flat, got_flat, N, N);
    end

    // ================================================================
    //  RANDOM SWEEP — 5000 tests
    //
    //  6 modes cycled evenly (t % 6):
    //    0=TYPE0-normal  1=TYPE0-transpose
    //    2=TYPE1-normal  3=TYPE1-transpose
    //    4=TYPE2-normal  5=TYPE2-transpose
    //
    //  Seed: fixed default 0xDEADBEEF for reproducibility.
    //  Override with +SEED=<hex> plusarg.
    //  On any failure the seed + iteration is printed for replay.
    // ================================================================
    $display("\n================================================");
    $display("  RANDOM SWEEP: 5000 tests (modes 0-5 cycled)");
    $display("================================================");

    // Seed — $value$plusargs with %d reads a decimal integer,
    // which is safe across all tool versions.
    rand_seed = 32'hDEAD_BEEF;
    if ($value$plusargs("SEED=%d", rand_seed_tmp))
        rand_seed = rand_seed_tmp;
    $srandom(rand_seed);
    $display("  Seed = 0x%08X  (rerun with +SEED=%0d)", rand_seed, rand_seed);

    for (rand_t = 0; rand_t < 5000; rand_t = rand_t + 1) begin

        rand_mode = rand_t % 6;

        if (rand_t % 1000 == 0 && rand_t > 0)
            $display("  ... %0d / 5000", rand_t);

        case (rand_mode)

            // ── TYPE 0 NORMAL ─────────────────────────────────────
            0 : begin
                foreach (rA0[i,k]) rA0[i][k] = 8'($urandom_range(0,255));
                foreach (rW0[r,c]) rW0[r][c] = 8'($urandom_range(0,255));
                golden_t0(rA0, rW0, rE0);
                reset_dut();
                run_type0(rA0, rW0, rC0);
                while (busy) tick;
                for (rand_i=0;rand_i<N;rand_i++) for (rand_j=0;rand_j<TOTAL;rand_j++) begin
                    rand_exp[rand_i*TOTAL+rand_j] = rE0[rand_i][rand_j];
                    rand_got[rand_i*TOTAL+rand_j] = rC0[rand_i][rand_j];
                end
                rand_label = $sformatf("RAND_%04d_t0_norm", rand_t);
                check_result(rand_label, rand_exp, rand_got, N, TOTAL);
            end

            // ── TYPE 0 TRANSPOSE ──────────────────────────────────
            1 : begin
                foreach (rA0[i,k]) rA0[i][k] = 8'($urandom_range(0,255));
                foreach (rW0[r,c]) rW0[r][c] = 8'($urandom_range(0,255));
                golden_t0(rA0, rW0, rE0);
                reset_dut();
                run_type0_tr(rA0, rW0, rC0);
                while (busy) tick;
                for (rand_i=0;rand_i<N;rand_i++) for (rand_j=0;rand_j<TOTAL;rand_j++) begin
                    rand_exp[rand_i*TOTAL+rand_j] = rE0[rand_i][rand_j];
                    rand_got[rand_i*TOTAL+rand_j] = rC0[rand_i][rand_j];
                end
                rand_label = $sformatf("RAND_%04d_t0_tr", rand_t);
                check_result(rand_label, rand_exp, rand_got, N, TOTAL);
            end

            // ── TYPE 1 NORMAL ─────────────────────────────────────
            2 : begin
                foreach (rA1[i,k]) rA1[i][k] = 8'($urandom_range(0,255));
                foreach (rW1[r,c]) rW1[r][c] = 8'($urandom_range(0,255));
                golden_t1(rA1, rW1, rE1);
                reset_dut();
                run_type1(rA1, rW1, rC1);
                while (busy) tick;
                for (rand_i=0;rand_i<N;rand_i++) for (rand_j=0;rand_j<2*N;rand_j++) begin
                    rand_exp[rand_i*2*N+rand_j] = rE1[rand_i][rand_j];
                    rand_got[rand_i*2*N+rand_j] = rC1[rand_i][rand_j];
                end
                rand_label = $sformatf("RAND_%04d_t1_norm", rand_t);
                check_result(rand_label, rand_exp, rand_got, N, 2*N);
            end

            // ── TYPE 1 TRANSPOSE ──────────────────────────────────
            3 : begin
                foreach (rA1[i,k]) rA1[i][k] = 8'($urandom_range(0,255));
                foreach (rW1[r,c]) rW1[r][c] = 8'($urandom_range(0,255));
                golden_t1(rA1, rW1, rE1);
                reset_dut();
                run_type1_tr(rA1, rW1, rC1);
                while (busy) tick;
                for (rand_i=0;rand_i<N;rand_i++) for (rand_j=0;rand_j<2*N;rand_j++) begin
                    rand_exp[rand_i*2*N+rand_j] = rE1[rand_i][rand_j];
                    rand_got[rand_i*2*N+rand_j] = rC1[rand_i][rand_j];
                end
                rand_label = $sformatf("RAND_%04d_t1_tr", rand_t);
                check_result(rand_label, rand_exp, rand_got, N, 2*N);
            end

            // ── TYPE 2 NORMAL ─────────────────────────────────────
            4 : begin
                foreach (rA2[i,k]) rA2[i][k] = 8'($urandom_range(0,255));
                foreach (rW2[r,c]) rW2[r][c] = 8'($urandom_range(0,255));
                golden_t2(rA2, rW2, rE2);
                reset_dut();
                run_type2(rA2, rW2, rC2);
                while (busy) tick;
                for (rand_i=0;rand_i<N;rand_i++) for (rand_j=0;rand_j<N;rand_j++) begin
                    rand_exp[rand_i*N+rand_j] = rE2[rand_i][rand_j];
                    rand_got[rand_i*N+rand_j] = rC2[rand_i][rand_j];
                end
                rand_label = $sformatf("RAND_%04d_t2_norm", rand_t);
                check_result(rand_label, rand_exp, rand_got, N, N);
            end

            // ── TYPE 2 TRANSPOSE ──────────────────────────────────
            5 : begin
                foreach (rA2[i,k]) rA2[i][k] = 8'($urandom_range(0,255));
                foreach (rW2[r,c]) rW2[r][c] = 8'($urandom_range(0,255));
                golden_t2(rA2, rW2, rE2);
                reset_dut();
                run_type2_tr(rA2, rW2, rC2);
                while (busy) tick;
                for (rand_i=0;rand_i<N;rand_i++) for (rand_j=0;rand_j<N;rand_j++) begin
                    rand_exp[rand_i*N+rand_j] = rE2[rand_i][rand_j];
                    rand_got[rand_i*N+rand_j] = rC2[rand_i][rand_j];
                end
                rand_label = $sformatf("RAND_%04d_t2_tr", rand_t);
                check_result(rand_label, rand_exp, rand_got, N, N);
            end

            default : ;
        endcase

    end // for rand_t

    $display("  ... 5000 / 5000 done.");
    $display("  Random sweep complete  (seed=0x%08X)", rand_seed);

    // ── Summary ───────────────────────────────────────────────────
    $display("");
    $display("================================================");
    $display("  TOTAL RESULTS:  %0d PASS    %0d FAIL", pass_cnt, fail_cnt);
    $display("  (10 directed + 5000 random = 5010 tests)");
    $display("================================================");
    if (fail_cnt == 0)
        $display("  >>> ALL TESTS PASSED <<<");
    else
        $display("  >>> FAILURES — see mismatches above <<<");
    $finish;
end

// ── Watchdog ──────────────────────────────────────────────────────
initial begin
    // 5010 tests x ~76 ticks/test x 10 ns/tick, with generous headroom
    #(5015 * (1 + N + N + (N+1) + N + 10) * 10);
    $display("TIMEOUT — simulation hung");
    $finish;
end

endmodule