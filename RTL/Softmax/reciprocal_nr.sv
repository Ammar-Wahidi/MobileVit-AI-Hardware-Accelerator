module reciprocal_nr
#(
    parameter DATA_WIDTH = 32,
    parameter FRAC_BITS  = 16
)
(
    input  logic signed [DATA_WIDTH-1:0] x,
    output logic signed [DATA_WIDTH-1:0] y
);

    always_comb begin
        if (x != 0) begin
            // Compute (2^(2*FRAC_BITS)) / x using 64-bit intermediate
            y = ( (64'(1) << FRAC_BITS) * (1 << FRAC_BITS) ) / x;
        end else begin
            y = 0;
        end
    end

endmodule