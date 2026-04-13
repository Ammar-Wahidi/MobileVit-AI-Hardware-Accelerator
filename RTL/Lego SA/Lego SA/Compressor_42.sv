// ================================================================
//  compressor_42 — Parameterised 4:2 Carry-Save Compressor
//
//  Reduces four DATA_W-wide inputs to two (DATA_W+1)-wide outputs
//  (sum, carry) using a bit-sliced two-FA chain.
//
//  Caller performs the final CPA:
//    result = sum + (carry << 1)
//
//  Per bit-slice b:
//
//    i0[b] ─┐
//    i1[b] ─┤ FA1 → s1[b], c1[b] ────────────► carry[b] = c1[b]
//    i2[b] ─┘
//                      s1[b] ─┐
//                      i3[b] ─┤ FA2 → fa2_sum[b], fa2_cout[b] → Cin of b+1
//                    cin_b  ─┘
//
//  carry[DATA_W]   = 0                  (FA1 carry chain never overflows)
//  sum  [DATA_W]   = fa2_cout[DATA_W-1] (absorb final column's FA2 cout)
//
//  Parameters:
//    DATA_W : operand bit-width (default 32)
// ================================================================

module Compressor_42 #(
    parameter int DATA_W = 32
)(
    input  logic [DATA_W-1:0] i0,
    input  logic [DATA_W-1:0] i1,
    input  logic [DATA_W-1:0] i2,
    input  logic [DATA_W-1:0] i3,

    output logic [DATA_W:0]   sum,    // carry-save sum   (weight-1 vector)
    output logic [DATA_W:0]   carry   // carry-save carry (weight-2 vector)
);

// ── Internal signals ──────────────────────────────────────────────
logic [DATA_W-1:0] s1;         // FA1 sum   bits
logic [DATA_W-1:0] c1;         // FA1 carry bits → carry output
logic [DATA_W-1:0] fa2_sum;    // FA2 sum   bits → lower DATA_W bits of sum
logic [DATA_W-1:0] fa2_cout;   // FA2 carry bits → chained Cin + MSB of sum

// ── Bit-sliced two-FA chain ───────────────────────────────────────
genvar b;
generate
    for (b = 0; b < DATA_W; b++) begin : CSA_SLICE

        // FA1 — compress i0, i1, i2
        assign s1[b]        =  i0[b] ^ i1[b] ^ i2[b];
        assign c1[b]        = (i0[b] & i1[b])
                            | (i1[b] & i2[b])
                            | (i0[b] & i2[b]);

        // FA2 — compress s1, i3, Cin
        //   Cin for bit 0 is 0 (no previous column carry)
        logic cin_b;
        assign cin_b        = (b == 0) ? 1'b0 : fa2_cout[b-1];

        assign fa2_sum [b]  =  s1[b]  ^ i3[b]  ^ cin_b;
        assign fa2_cout[b]  = (s1[b]  & i3[b])
                            | (i3[b]  & cin_b)
                            | (s1[b]  & cin_b);

    end
endgenerate

// ── Output assembly ───────────────────────────────────────────────
// carry : FA1 carry bits, weight-2.
//         Guard MSB = 0 (FA1 carry chain cannot overflow DATA_W+1 bits).
assign carry = { 1'b0,               c1      };

// sum   : FA2 sum bits, weight-1.
//         MSB = fa2_cout of the top column, so the caller's CPA
//         can absorb any carry that propagates out of bit DATA_W-1.
assign sum   = { fa2_cout[DATA_W-1], fa2_sum };

endmodule