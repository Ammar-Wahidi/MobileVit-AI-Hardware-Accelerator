// ================================================================
//  Lego_SA — LEGO Reconfigurable Systolic Array
//
//  Four 16×16 SA_NxN_top tiles controlled by a single Lego_CU.
//  SA_NxN_top tiles are pure datapaths — no internal SA_CU.
//  Lego_CU drives load_w, valid_out, and busy for the entire system.
//
//  ── Tile layout ────────────────────────────────────────────────
//
//    ┌──────────┬──────────┐
//    │  RU (SA0)│  LU (SA1)│  ← top row
//    ├──────────┼──────────┤
//    │  RD (SA2)│  LD (SA3)│  ← bottom row
//    └──────────┴──────────┘
//     left col   right col
//
//  ── Shape types ────────────────────────────────────────────────
//
//    TYPE 0 — (16 rows × 64 cols)
//      W: 16×64,  A: 16×16,  C: 16×64
//      All tiles compute different column slices of the same W.
//      Same act broadcast to all 4 tiles.
//      Output: concatenate {RU, LU, RD, LD} = 64 elements.
//
//    TYPE 1 — (32 rows × 32 cols)
//      W: 32×32,  A: 16×32,  C: 16×32
//      RU/LU hold top half of weight rows; RD/LD hold bottom half.
//      Top act half → RU, LU;  bottom act half → RD, LD.
//      Output: RU+RD (left cols), LU+LD (right cols) = 32 elements.
//
//    TYPE 2 — (64 rows × 16 cols)
//      W: 64×16,  A: 16×64,  C: 16×16
//      Each tile holds its own 16-row weight slice.
//      Each tile gets its own 16-element act slice.
//      Output: RU+RD+LU+LD (4-way add) = 16 elements.
//
//  ── Weight bus packing (weight_in always 4×N_TILE wide) ────────
//
//    weight_in[0    : N-1  ] → RU
//    weight_in[N    : 2N-1 ] → LU
//    weight_in[2N   : 3N-1 ] → RD
//    weight_in[3N   : 4N-1 ] → LD
//
//    Caller interleaves rows per type (see comments in weight routing).
//
//  ── Activation bus packing (act_in always 4×N_TILE wide) ───────
//
//    TYPE 0: act_in[0:N-1] broadcast → RU, LU, RD, LD
//    TYPE 1: act_in[0:N-1] → RU, LU;  act_in[N:2N-1] → RD, LD
//    TYPE 2: act_in[0:N-1]→RU  [N:2N-1]→RD  [2N:3N-1]→LU  [3N:4N-1]→LD
//
//  ── Output bus (psum_out always 4×N_TILE wide) ─────────────────
//
//    TYPE 0: psum_out[0:N-1]=RU  [N:2N-1]=LU  [2N:3N-1]=RD  [3N:4N-1]=LD
//    TYPE 1: psum_out[0:N-1]=RU+RD  [N:2N-1]=LU+LD  [2N:4N-1]=0
//    TYPE 2: psum_out[0:N-1]=RU+RD+LU+LD  [N:4N-1]=0
//
//  ── Timing (from valid_in assertion) ───────────────────────────
//
//    LOAD_W : N_TILE cycles  valid_in=1, weight_in valid
//    FEED_A : N_TILE cycles  valid_in=1, act_in valid
//    DRAIN  : N_TILE-1 cycles  autonomous
//    OUTPUT : N_TILE cycles  autonomous, valid_out=1
//
// ================================================================

module Lego_SA #(
    parameter DATA_W     = 8 ,
    parameter DATA_W_OUT = 32,
    parameter N_TILE     = 16
)(
    input  logic                       clk          ,
    input  logic                       rst_n        ,

    // ── Control ─────────────────────────────────────────────────
    input  logic                       valid_in     ,
    input  logic [1:0]                 lego_type    ,
    input  logic [7:0]                 y_input_size ,
    input  logic                       transpose_en ,

    // ── Data inputs (4×N_TILE wide) ─────────────────────────────
    input  logic [DATA_W-1:0]          act_in    [4*N_TILE],
    input  logic [DATA_W-1:0]          weight_in [4*N_TILE],

    // ── Outputs ─────────────────────────────────────────────────
    output logic [DATA_W_OUT-1:0]      psum_out  [4*N_TILE],
    output logic                       valid_out            ,
    output logic                       busy
);

// ── Local parameters ──────────────────────────────────────────────
localparam TOTAL_W = 4 * N_TILE;

// ── Per-tile data signals ─────────────────────────────────────────
logic [DATA_W-1:0]     tile_act_RU [N_TILE];
logic [DATA_W-1:0]     tile_act_LU [N_TILE];
logic [DATA_W-1:0]     tile_act_RD [N_TILE];
logic [DATA_W-1:0]     tile_act_LD [N_TILE];

logic [DATA_W-1:0]     tile_w_RU   [N_TILE];
logic [DATA_W-1:0]     tile_w_LU   [N_TILE];
logic [DATA_W-1:0]     tile_w_RD   [N_TILE];
logic [DATA_W-1:0]     tile_w_LD   [N_TILE];

logic [DATA_W_OUT-1:0] psum_RU     [N_TILE];
logic [DATA_W_OUT-1:0] psum_LU     [N_TILE];
logic [DATA_W_OUT-1:0] psum_RD     [N_TILE];
logic [DATA_W_OUT-1:0] psum_LD     [N_TILE];

// ── Control signal from Lego_CU ───────────────────────────────────
// load_w: broadcast to all 4 tiles simultaneously.
// Replaces the per-tile SA_CU that previously generated it internally.
logic load_w_cu;

// ── Output adder wires ────────────────────────────────────────────
logic [DATA_W_OUT-1:0] psum_add_left  [N_TILE];  // RU + RD
logic [DATA_W_OUT-1:0] psum_add_right [N_TILE];  // LU + LD
logic [DATA_W_OUT-1:0] psum_add_all   [N_TILE];  // RU + RD + LU + LD

genvar g;
generate
    for (g = 0; g < N_TILE; g++) begin : OUTPUT_ADD
        assign psum_add_left [g] = psum_RU[g] + psum_RD[g];
        assign psum_add_right[g] = psum_LU[g] + psum_LD[g];
        assign psum_add_all  [g] = psum_RU[g] + psum_RD[g]
                                 + psum_LU[g] + psum_LD[g];
    end
endgenerate

// ================================================================
//  ① Lego_CU — The ONE AND ONLY control unit
//
//  Generates load_w, valid_out, and busy for the entire system.
//  No other control logic exists anywhere in the design.
// ================================================================
Lego_CU #(
    .N_TILE (N_TILE)
) u_lego_cu (
    .clk          (clk          ),
    .rst_n        (rst_n        ),
    .valid_in     (valid_in     ),
    .lego_type    (lego_type    ),
    .y_input_size (y_input_size ),
    .load_w       (load_w_cu    ),
    .valid_out    (valid_out    ),
    .busy         (busy         )
);

// ================================================================
//  ② Weight routing
//
//  weight_in bus is sliced into 4 equal N_TILE-wide sections.
//  Each section goes to one tile's weight_in port.
//  Routing is fixed — the caller interleaves rows per type:
//
//    TYPE 0 — load cycle k:
//      weight_in[0:N-1]   = W[k][0:N-1]       → RU (col slice 0)
//      weight_in[N:2N-1]  = W[k][N:2N-1]      → LU (col slice 1)
//      weight_in[2N:3N-1] = W[k][2N:3N-1]     → RD (col slice 2)
//      weight_in[3N:4N-1] = W[k][3N:4N-1]     → LD (col slice 3)
//
//    TYPE 1 — load cycle k:
//      weight_in[0:N-1]   = W[k][0:N-1]        → RU (row k,   left cols)
//      weight_in[N:2N-1]  = W[k][N:2N-1]       → LU (row k,   right cols)
//      weight_in[2N:3N-1] = W[k+N][0:N-1]      → RD (row k+N, left cols)
//      weight_in[3N:4N-1] = W[k+N][N:2N-1]     → LD (row k+N, right cols)
//
//    TYPE 2 — load cycle k:
//      weight_in[0:N-1]   = W[k   ][0:N-1]     → RU (rows 0..N-1)
//      weight_in[N:2N-1]  = W[k+N ][0:N-1]     → RD (rows N..2N-1)
//      weight_in[2N:3N-1] = W[k+2N][0:N-1]     → LU (rows 2N..3N-1)
//      weight_in[3N:4N-1] = W[k+3N][0:N-1]     → LD (rows 3N..4N-1)
// ================================================================
always_comb begin
    tile_w_RU = '{default:'0};
    tile_w_LU = '{default:'0};
    tile_w_RD = '{default:'0};
    tile_w_LD = '{default:'0};

    if (load_w_cu) begin
        tile_w_RU = weight_in[0        : N_TILE-1  ];
        tile_w_LU = weight_in[N_TILE   : 2*N_TILE-1];
        tile_w_RD = weight_in[2*N_TILE : 3*N_TILE-1];
        tile_w_LD = weight_in[3*N_TILE : TOTAL_W-1 ];
    end
end

// ================================================================
//  ③ Activation routing
//
//  TYPE 0: broadcast act_in[0:N-1] to all 4 tiles
//  TYPE 1: act_in[0:N-1] → RU,LU;  act_in[N:2N-1] → RD,LD
//  TYPE 2: four independent 16-element slices to four tiles
//
//  Routing is active only during FEED_A (load_w_cu=0 and valid_in=1).
//  During LOAD_W, act inputs are zeroed (don't-care but safe).
// ================================================================
always_comb begin
    tile_act_RU = '{default:'0};
    tile_act_LU = '{default:'0};
    tile_act_RD = '{default:'0};
    tile_act_LD = '{default:'0};

    if (!load_w_cu && valid_in) begin
        case (lego_type)

            // TYPE 0: 16×64 — same act broadcast to all 4 tiles
            2'd0 : begin
                tile_act_RU = act_in[0       : N_TILE-1  ];
                tile_act_LU = act_in[0       : N_TILE-1  ];
                tile_act_RD = act_in[0       : N_TILE-1  ];
                tile_act_LD = act_in[0       : N_TILE-1  ];
            end

            // TYPE 1: 32×32 — top/bottom split
            2'd1 : begin
                tile_act_RU = act_in[0      : N_TILE-1  ];
                tile_act_LU = act_in[0      : N_TILE-1  ];
                tile_act_RD = act_in[N_TILE : 2*N_TILE-1];
                tile_act_LD = act_in[N_TILE : 2*N_TILE-1];
            end

            // TYPE 2: 64×16 — 4-way independent split
            2'd2 : begin
                tile_act_RU = act_in[0        : N_TILE-1  ];
                tile_act_RD = act_in[N_TILE   : 2*N_TILE-1];
                tile_act_LU = act_in[2*N_TILE : 3*N_TILE-1];
                tile_act_LD = act_in[3*N_TILE : TOTAL_W-1 ];
            end

            default : begin /* all zeros from default above */ end

        endcase
    end
end

// ================================================================
//  ④ Output mux
//
//  Selects and combines tile outputs according to lego_type.
//  Always combinatorial from tile psum outputs.
//  Meaningful only while valid_out=1 (caller should gate on that).
// ================================================================
always_comb begin
    psum_out = '{default:'0};

    case (lego_type)

        // TYPE 0: direct concatenation — 64 valid elements
        2'd0 : begin
            psum_out[0        : N_TILE-1  ] = psum_RU;
            psum_out[N_TILE   : 2*N_TILE-1] = psum_LU;
            psum_out[2*N_TILE : 3*N_TILE-1] = psum_RD;
            psum_out[3*N_TILE : TOTAL_W-1 ] = psum_LD;
        end

        // TYPE 1: element-wise add paired tiles — 32 valid elements
        // [0:N-1]  = RU+RD  (left  column block, top+bottom rows contribution)
        // [N:2N-1] = LU+LD  (right column block)
        2'd1 : begin
            psum_out[0      : N_TILE-1  ] = psum_add_left;
            psum_out[N_TILE : 2*N_TILE-1] = psum_add_right;
        end

        // TYPE 2: 4-way element-wise add — 16 valid elements
        // Each tile contributes partial sums over its 16-row weight slice.
        // Sum gives the complete dot product result.
        2'd2 : begin
            psum_out[0 : N_TILE-1] = psum_add_all;
        end

        default : psum_out = '{default:'0};

    endcase
end

// ================================================================
//  ⑤ SA_NxN_top tile instances — pure datapaths
//
//  load_w_cu (from Lego_CU) is broadcast to ALL 4 tiles.
//  No tile has its own SA_CU — Lego_CU is the single authority.
// ================================================================

// RU — top-left tile
L_SA_NxN_top #(
    .DATA_W    (DATA_W    ),
    .DATA_W_OUT(DATA_W_OUT),
    .N_SIZE    (N_TILE    )
) u_SA_RU (
    .clk         (clk         ),
    .rst_n       (rst_n       ),
    .act_in      (tile_act_RU ),
    .weight_in   (tile_w_RU   ),
    .transpose_en(transpose_en),
    .load_w      (load_w_cu   ),
    .psum_out    (psum_RU     )
);

// LU — top-right tile
L_SA_NxN_top #(
    .DATA_W    (DATA_W    ),
    .DATA_W_OUT(DATA_W_OUT),
    .N_SIZE    (N_TILE    )
) u_SA_LU (
    .clk         (clk         ),
    .rst_n       (rst_n       ),
    .act_in      (tile_act_LU ),
    .weight_in   (tile_w_LU   ),
    .transpose_en(transpose_en),
    .load_w      (load_w_cu   ),
    .psum_out    (psum_LU     )
);

// RD — bottom-left tile
L_SA_NxN_top #(
    .DATA_W    (DATA_W    ),
    .DATA_W_OUT(DATA_W_OUT),
    .N_SIZE    (N_TILE    )
) u_SA_RD (
    .clk         (clk         ),
    .rst_n       (rst_n       ),
    .act_in      (tile_act_RD ),
    .weight_in   (tile_w_RD   ),
    .transpose_en(transpose_en),
    .load_w      (load_w_cu   ),
    .psum_out    (psum_RD     )
);

// LD — bottom-right tile
L_SA_NxN_top #(
    .DATA_W    (DATA_W    ),
    .DATA_W_OUT(DATA_W_OUT),
    .N_SIZE    (N_TILE    )
) u_SA_LD (
    .clk         (clk         ),
    .rst_n       (rst_n       ),
    .act_in      (tile_act_LD ),
    .weight_in   (tile_w_LD   ),
    .transpose_en(transpose_en),
    .load_w      (load_w_cu   ),
    .psum_out    (psum_LD     )
);

endmodule