module swish_array #(
    parameter int N = 4,             // Number of parallel elements
    parameter int WIDTH = 16,        // Data width per element
    parameter int FRACT_BITS = 8     // Q-notation fractional bits
)(
    // Unpacked Arrays: The array dimension [0:N-1] comes AFTER the port name
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  logic signed [WIDTH-1:0] x [0:N-1],
    output logic signed [WIDTH-1:0] y [0:N-1],
    output logic valid_out
);

    logic signed [WIDTH-1:0] x_reg  [0:N-1] ;
    logic signed [WIDTH-1:0] y_comb [0:N-1] ;

    genvar j;
    generate
        for(j=0;j < N; j=j+1)
        begin : gen_input_regs
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                begin
                    x_reg[j] <= '0 ;
                end
                else if (valid_in)
                begin
                    x_reg[j] <= x[j] ;
                end
            end
        end
    endgenerate
    
    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : gen_swish_lanes
            
            // Instantiate the scalar swish module for each lane
            swish #(
                .WIDTH(WIDTH),
                .FRACT_BITS(FRACT_BITS)
            ) swish_inst (
                .x(x_reg[i]),
                .y(y_comb[i])
            );
            
        end
    endgenerate

    genvar k;
    generate
        for (k=0; k < N; k = k + 1 ) begin : gen_output_regs
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    y[k] <= '0;
                else if (valid_in)
                    y[k] <= y_comb[k];
            end
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            valid_out <= 1'b1 ;
        else
            valid_out <= valid_in ;
    end

endmodule