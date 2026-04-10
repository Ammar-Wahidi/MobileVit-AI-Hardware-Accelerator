module ELEMENTS_SUM_2 #(
    parameter DATA_WIDTH = 64,
    parameter EMBED_DIM  = 32 // Must be a power of 2 for this simple version
)(
    input  wire                                               clk                           ,
    input  wire                                               rst_n                         ,
    input  wire                                               sum_en_2                      ,
    input  wire signed [DATA_WIDTH-1:0]                       mul_in [0:EMBED_DIM-1]        ,
    output wire signed [DATA_WIDTH+($clog2(EMBED_DIM))-1:0]   sum_2_out                     ,
    output reg                                                sum_2_out_valid
);

    // Localparam to calculate intermediate stages
    localparam STAGES = $clog2(EMBED_DIM);

    // Recursive logic for the adder tree
    generate
        if (EMBED_DIM == 1) begin : base_case
            assign sum_2_out = mul_in[0];
            
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) sum_2_out_valid <= 1'b0;
                else        sum_2_out_valid <= sum_en_2;
            end
        end 
        else begin : recursive_step
            // Internal wires for the current stage results
            wire signed [DATA_WIDTH+1-1:0] stage_sums [0:(EMBED_DIM/2)-1];
            reg  signed [DATA_WIDTH+1-1:0] stage_regs [0:(EMBED_DIM/2)-1];
            reg                            stage_en;

            // Pairwise addition
            genvar i;
            for (i = 0; i < EMBED_DIM/2; i = i + 1) begin : adders
                assign stage_sums[i] = mul_in[2*i] + mul_in[2*i+1];
            end

            // Pipeline Registers
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    stage_en <= 1'b0;
                end else begin
                    stage_en <= sum_en_2;
                    for (int j = 0; j < EMBED_DIM/2; j = j + 1) begin
                        stage_regs[j] <= stage_sums[j];
                    end
                end
            end

            // Instantiate the next level of the tree
            ELEMENTS_SUM_2 #(
                .DATA_WIDTH (DATA_WIDTH + 1),
                .EMBED_DIM  (EMBED_DIM / 2)
            ) sub_tree (
                .clk        (clk),
                .rst_n      (rst_n),
                .sum_en_2   (stage_en),
                .mul_in  (stage_regs),
                .sum_2_out    (sum_2_out),
                .sum_2_out_valid  (sum_2_out_valid)
            );
        end
    endgenerate

endmodule