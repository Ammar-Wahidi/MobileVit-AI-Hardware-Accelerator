module swish_array #(
    parameter int N = 4,             // Number of parallel elements
    parameter int WIDTH = 16,        // Data width per element
    parameter int FRACT_BITS = 8     // Q-notation fractional bits
)(
    // Unpacked Arrays: The array dimension [0:N-1] comes AFTER the port name
    input  wire signed [WIDTH-1:0] x [0:N-1],
    output wire signed [WIDTH-1:0] y [0:N-1]
);

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : gen_swish_lanes
            
            // Instantiate the scalar swish module for each lane
            swish #(
                .WIDTH(WIDTH),
                .FRACT_BITS(FRACT_BITS)
            ) swish_inst (
                .x(x[i]),
                .y(y[i])
            );
            
        end
    endgenerate

endmodule