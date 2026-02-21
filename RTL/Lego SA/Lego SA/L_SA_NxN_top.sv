// ================================================================
//  L_SA_NxN_top — Datapath-Only Systolic Array Tile
//
//  This is the tile module used inside Lego_SA. Unlike SA_NxN_top,
//  it contains no control unit. load_w is driven externally by
//  the single shared Lego_CU, which controls all four tiles at
//  once. valid_in, valid_out, and busy do not exist here.
//
//  Sub-modules:
//    TRSRL  : Delays act_in[k] by k cycles (activation skew).
//    SA_NxN : N x N weight-stationary PE mesh.
//    TRSDL  : Realigns psum columns so all N results appear
//             simultaneously on psum_out (output de-skew).
//
//  Parameters:
//    DATA_W     : activation / weight bit-width  (default 8)
//    DATA_W_OUT : accumulator bit-width          (default 32)
//    N_SIZE     : tile dimension                 (default 16)
//
//  Control interface:
//    load_w is the only control input. It comes directly from
//    Lego_CU and is shared with all other tiles in the array.
//    When load_w=1, PEs latch weight_in. When load_w=0, PEs
//    accumulate act_in*weight.
//
//  Output de-skew detail:
//    Same mirror-TRSDL-mirror scheme as SA_NxN_top.
//    psum_to_dl[i] = psum[N-1-i]  (reverse before TRSDL)
//    psum_out[i]   = psum_dl[N-1-i] (reverse after TRSDL)
//
//  Hierarchy:
//    L_SA_NxN_top  (one of four tiles inside Lego_SA)
//    +-- TRSRL   (activation skew)
//    +-- SA_NxN  (datapath, N^2 PEs)
//    +-- TRSDL   (output de-skew)
//
// ================================================================

module L_SA_NxN_top #(
    parameter DATA_W     = 8,
    parameter DATA_W_OUT = 32,
    parameter N_SIZE     = 16
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // Data inputs
    input  logic [DATA_W-1:0]       act_in    [N_SIZE],  // activation rows for this tile
    input  logic [DATA_W-1:0]       weight_in [N_SIZE],  // weight rows for this tile
    input  logic                    transpose_en,        // 0 = load from bottom, 1 = load from right

    // Control (driven by Lego_CU, shared with all tiles)
    input  logic                    load_w,              // 1 = latch weights, 0 = accumulate

    // Result
    output logic [DATA_W_OUT-1:0]   psum_out  [N_SIZE]  // de-skewed partial sums for this tile
);

// ── Internal wires ────────────────────────────────────────────────
logic [DATA_W-1:0]    act_skewed [N_SIZE];   // TRSRL output: diagonally skewed activations
logic [DATA_W_OUT-1:0] psum      [N_SIZE];   // SA_NxN output: raw (still skewed) partial sums
logic [DATA_W_OUT-1:0] psum_to_dl[N_SIZE];   // column-mirrored psums fed into TRSDL
logic [DATA_W_OUT-1:0] psum_dl   [N_SIZE];   // TRSDL output: de-skewed partial sums

// ── Activation skew — TRSRL ───────────────────────────────────────
// Delays act_in[k] by k cycles so activations enter the PE array
// on the correct diagonal cycle for weight-stationary computation.
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
// PEs hold their weights stationary throughout FEED_A.
// load_w comes directly from the external Lego_CU.
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
// Mirror the column order before TRSDL so that the column with
// the greatest skew (N-1) maps to the TRSDL lane with the most
// registers. Mirror again after TRSDL to restore natural order.
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