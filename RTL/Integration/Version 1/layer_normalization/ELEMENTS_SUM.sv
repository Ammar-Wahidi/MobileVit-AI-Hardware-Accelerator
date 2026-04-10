module ELEMENTS_SUM #(
    parameter DATA_WIDTH = 32,
    parameter EMBED_DIM  = 32 // Must be a power of 2 for this simple version
)(
    input  wire                                               clk                           ,
    input  wire                                               rst_n                         ,
    input  wire                                               sum_en_1                      ,
    input  wire signed [DATA_WIDTH-1:0]                       activation_in [0:EMBED_DIM-1] ,
    output wire signed [DATA_WIDTH+($clog2(EMBED_DIM))-1:0]   sum_1_out                     ,
    output reg                                                sum_1_out_valid
);

    // Localparam to calculate intermediate stages
    localparam STAGES = $clog2(EMBED_DIM);

    // Recursive logic for the adder tree
    generate
        if (EMBED_DIM == 1) begin : base_case
            assign sum_1_out = activation_in[0];
            
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) sum_1_out_valid <= 1'b0;
                else        sum_1_out_valid <= sum_en_1;
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
                assign stage_sums[i] = activation_in[2*i] + activation_in[2*i+1];
            end

            // Pipeline Registers
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    stage_en <= 1'b0;
                end else begin
                    stage_en <= sum_en_1;
                    for (int j = 0; j < EMBED_DIM/2; j = j + 1) begin
                        stage_regs[j] <= stage_sums[j];
                    end
                end
            end

            // Instantiate the next level of the tree
            ELEMENTS_SUM #(
                .DATA_WIDTH (DATA_WIDTH + 1),
                .EMBED_DIM  (EMBED_DIM / 2)
            ) sub_tree (
                .clk        (clk),
                .rst_n      (rst_n),
                .sum_en_1   (stage_en),
                .activation_in  (stage_regs),
                .sum_1_out    (sum_1_out),
                .sum_1_out_valid  (sum_1_out_valid)
            );
        end
    endgenerate

endmodule