module exp
#(
    parameter DATA_WIDTH = 32,
    parameter FRAC_BITS  = 16
)
(
    input  logic signed [DATA_WIDTH-1:0] x,
    output logic signed [DATA_WIDTH-1:0] y
);

    // LUT for e^x for x = -8.0 to 0.0 step 0.125 (Q16.16 format)
    // Values are pre‑computed for FRAC_BITS = 16.
    localparam signed [DATA_WIDTH-1:0] LUT [0:64] = '{
        32'd22, 32'd25, 32'd28, 32'd32, 32'd37, 32'd42, 32'd47, 32'd54,
        32'd60, 32'd68, 32'd77, 32'd88, 32'd100, 32'd113, 32'd129, 32'd146,
        32'd163, 32'd185, 32'd210, 32'd238, 32'd271, 32'd308, 32'd350, 32'd398,
        32'd442, 32'd502, 32'd570, 32'd648, 32'd737, 32'd837, 32'd952, 32'd1082,
        32'd1200, 32'd1364, 32'd1551, 32'd1763, 32'd2004, 32'd2277, 32'd2589, 32'd2943,
        32'd3263, 32'd3708, 32'd4214, 32'd4790, 32'd5379, 32'd6111, 32'd6944, 32'd7892,
        32'd8870, 32'd10080, 32'd11456, 32'd13021, 32'd14801, 32'd16822, 32'd19118, 32'd21729,
        32'd24107, 32'd27319, 32'd30955, 32'd35075, 32'd39749, 32'd45037, 32'd51036, 32'd57828,
        32'd65536   // x = 0.0
    };

    logic signed [DATA_WIDTH-1:0] x_clamped;
    logic signed [DATA_WIDTH-1:0] offset;
    logic [6:0] index;          // 0..64 (7 bits)
    logic [12:0] frac;          // lower 13 bits (step size = 2^13)
    logic signed [DATA_WIDTH-1:0] base;
    logic signed [DATA_WIDTH-1:0] correction;

    always_comb begin
        // Clamp x to the range covered by the LUT [-8, 0]
        if (x < - (8 << FRAC_BITS))
            x_clamped = - (8 << FRAC_BITS);
        else
            x_clamped = x;

        // Convert to positive offset from -8
        offset = x_clamped + (8 << FRAC_BITS);   // now in [0, 8<<16]

        // Extract table index and fractional part
        // Step size = 2^(FRAC_BITS-3) = 2^13 = 8192
        index = offset >>> 13;
        frac  = offset[12:0];     // lower 13 bits

        base = LUT[index];

        // Linear interpolation: y = base + (base * frac) / 2^FRAC_BITS
        correction = (base * signed'({1'b0, frac})) >>> FRAC_BITS;

        y = base + correction;

        // Safety clamp (should never be negative)
        if (y < 0)
            y = 0;
    end

endmodule