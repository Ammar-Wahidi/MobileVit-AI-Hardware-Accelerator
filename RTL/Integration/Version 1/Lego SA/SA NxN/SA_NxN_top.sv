// ================================================================
//  SA_NxN_top — Standalone N x N Systolic Array Top-Level
//
//  A self-contained, single-tile matrix multiply accelerator.
//  Integrates the control unit, activation skew, the systolic
//  array core, and the output de-skew into one module.
//
//  Sub-modules:
//    SA_CU   : FSM that sequences LOAD_W, FEED_A, DRAIN, OUTPUT.
//    TRSRL   : Triangular shift-right logic; delays act_in[k] by k
//              cycles so activations enter the array diagonally.
//    SA_NxN  : N x N weight-stationary PE mesh.
//    TRSDL   : Triangular shift-down logic; re-aligns the partial
//              sum columns so all N results appear simultaneously.
//
//  Parameters:
//    DATA_W     : activation / weight bit-width  (default 8)
//    DATA_W_OUT : accumulator bit-width          (default 32)
//    N_SIZE     : array dimension                (default 16)
//
//  Protocol:
//    1. Assert valid_in=1.
//    2. LOAD_W  (N_SIZE valid cycles) : drive weight_in each cycle.
//    3. FEED_A  (N_SIZE valid cycles) : drive act_in each cycle.
//    4. DRAIN   (N_SIZE-1 cycles)     : wait, no input needed.
//    5. OUTPUT  (N_SIZE cycles)       : read psum_out while valid_out=1.
//    6. Wait for busy=0 before starting the next matmul.
//
//  Output de-skew detail:
//    The SA produces psum[j] one cycle later than psum[j-1] because
//    column j's activations arrive one cycle later (TRSRL skew).
//    Before TRSDL: the column order is mirrored so that the column
//    needing the most delay maps to the lane with the most registers.
//    After  TRSDL: the mirror is applied again to restore the natural
//    column order on psum_out.
//
//  Hierarchy:
//    SA_NxN_top
//    +-- SA_CU   (control)
//    +-- TRSRL   (activation skew)
//    +-- SA_NxN  (datapath, N^2 PEs)
//    +-- TRSDL   (output de-skew)
//
// ================================================================

module SA_NxN_top #(
    parameter DATA_W     = 8,
    parameter DATA_W_OUT = 32,
    parameter N_SIZE     = 16
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // Data inputs
    input  logic [DATA_W-1:0]       act_in    [N_SIZE],  // activation rows (drive during FEED_A)
    input  logic [DATA_W-1:0]       weight_in [N_SIZE],  // weight rows     (drive during LOAD_W)
    input  logic                    transpose_en,        // 0 = load from bottom, 1 = load from right

    // Handshake
    input  logic                    valid_in,            // data-valid / matmul start trigger
    output logic                    valid_out,           // HIGH for N_SIZE cycles during OUTPUT
    output logic                    busy,                // HIGH while any phase is active

    // Result
    output logic [DATA_W_OUT-1:0]   psum_out  [N_SIZE]  // de-skewed output (valid when valid_out=1)
);

// ── Internal wires ────────────────────────────────────────────────
logic                   load_w;           // SA_CU → SA_NxN: weight-latch enable
logic [DATA_W-1:0]      act_skewed [N_SIZE];   // TRSRL output: skewed activations
logic [DATA_W_OUT-1:0]  psum       [N_SIZE];   // SA_NxN output: raw (still skewed) partial sums
logic [DATA_W_OUT-1:0]  psum_to_dl [N_SIZE];   // column-reversed partial sums fed into TRSDL
logic [DATA_W_OUT-1:0]  psum_dl    [N_SIZE];   // TRSDL output: de-skewed partial sums

// ── Control unit ──────────────────────────────────────────────────
// Generates load_w, valid_out, and busy for the entire module.
SA_CU #(
    .N_SIZE (N_SIZE)
) u_cu (
    .clk      (clk      ),
    .rst_n    (rst_n    ),
    .valid_in (valid_in ),
    .load_w   (load_w   ),
    .valid_out(valid_out),
    .busy     (busy     )
);

// ── Activation skew — TRSRL ───────────────────────────────────────
// act_in[k] is delayed by k cycles before entering the array.
// This staggers the activation wavefront to match the weight-
// stationary diagonal computation in SA_NxN.
TRSRL #(
    .DATAWIDTH (DATA_W ),
    .N_SIZE    (N_SIZE )
) u_trsrl (
    .clk    (clk       ),
    .rst_n  (rst_n     ),
    .act_in (act_in    ),
    .act_out(act_skewed)
);

// ── N x N systolic array ──────────────────────────────────────────
// Weight-stationary: PEs latch weight_in when load_w=1, then
// compute act*weight accumulations for N_SIZE cycles when load_w=0.
SA_NxN #(
    .DATA_W    (DATA_W    ),
    .DATA_W_OUT(DATA_W_OUT),
    .N_SIZE    (N_SIZE    )
) u_sa (
    .clk         (clk         ),
    .rst_n       (rst_n       ),
    .act_in      (act_skewed  ),
    .weight_in   (weight_in   ),
    .load_w      (load_w      ),
    .transpose_en(transpose_en),
    .psum_out    (psum        )
);

// ── Output de-skew — TRSDL ───────────────────────────────────────
// Step 1: reverse column order before TRSDL.
//         psum[N-1] (most delayed) maps to lane 0 (most registers).
// Step 2: TRSDL equalises all delays.
// Step 3: reverse column order again to restore natural order.
genvar i;
generate
    for (i = 0; i < N_SIZE; i++) begin : REORDER
        assign psum_to_dl[i] = psum[N_SIZE - 1 - i];
        assign psum_out[i]   = psum_dl[N_SIZE - 1 - i];
    end
endgenerate

TRSDL #(
    .DATAWIDTH (DATA_W_OUT),
    .N_SIZE    (N_SIZE    )
) u_trsdl (
    .clk     (clk       ),
    .rst_n   (rst_n     ),
    .psum_in (psum_to_dl),
    .psum_out(psum_dl   )
);

endmodule