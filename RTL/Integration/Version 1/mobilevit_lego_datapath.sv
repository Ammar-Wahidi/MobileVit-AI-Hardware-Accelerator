module mobilevit_lego_datapath #(
    parameter DATA_W     = 8,
    parameter DATA_W_OUT = 32,
    parameter SWISH_W    = 32,
    parameter FRACT_BITS = 1,
    parameter N_TILE     = 16,
    parameter TOTAL_W    = 4 * N_TILE // 64 elements
)(
    input  logic                        clk,
    input  logic                        rst_n,

    // Independent Parallel Control Signals
    input  logic                        valid_in1,    // 1 = Trigger Swish Path
    input  logic                        valid_in2,   // 1 = Trigger LayerNorm Path
    input  logic                        valid_in,    
    
    input  logic [1:0]                  lego_type,   
    input  logic [7:0]                  y_input_size, 
    input  logic                        transpose_en,
    input  logic                        vpu_add_en,   // 1 = Trigger Residual Add

    // Data Inputs (From SRAM)
    input  logic signed [DATA_W-1:0]    act_in       [0:TOTAL_W-1], 
    input  logic signed [DATA_W-1:0]    weight_in    [0:TOTAL_W-1], 
    input  logic signed [SWISH_W-1:0]   skip_conn_in [0:TOTAL_W-1], 
    input  logic signed [SWISH_W-1:0]   add_main_in  [0:TOTAL_W-1], 

    // Final Outputs
    output logic signed [SWISH_W-1:0]   final_out    [0:TOTAL_W-1],
    output logic                        final_valid_out,
    
    // Status
    output logic                        sa_valid_out, 
    output logic                        sa_busy       
);

    // ---------------------------------------------------------
    // 0. Setup & Trigger Logic
    // ---------------------------------------------------------
    logic sa_start;
    assign sa_start = valid_in | valid_in1 | valid_in2; // SA runs if either path is requested

    logic [DATA_W-1:0]     sa_act_unsigned    [0:TOTAL_W-1]; 
    logic [DATA_W-1:0]     sa_weight_unsigned [0:TOTAL_W-1]; 
    logic [DATA_W_OUT-1:0] sa_psum_unsigned   [0:TOTAL_W-1]; 
    logic signed [DATA_W_OUT-1:0] sa_psum     [0:TOTAL_W-1]; 

    genvar c;
    generate
        for (c = 0; c < TOTAL_W; c++) begin : TYPE_CASTING
            assign sa_act_unsigned[c]    = act_in[c]; 
            assign sa_weight_unsigned[c] = weight_in[c]; 
            assign sa_psum[c]            = $signed(sa_psum_unsigned[c]);
        end
    endgenerate

    // ---------------------------------------------------------
    // 1. Systolic Array (Core Engine)
    // ---------------------------------------------------------
    Lego_SA #(
        .DATA_W(DATA_W), 
        .DATA_W_OUT(DATA_W_OUT), 
        .Y_INPUT_SIZE(8), 
        .N_TILE(N_TILE) 
    ) u_lego_sa (
        .clk(clk), .rst_n(rst_n), .valid_in(sa_start),
        .lego_type(lego_type), .y_input_size(y_input_size), .transpose_en(transpose_en),
        .act_in(sa_act_unsigned), .weight_in(sa_weight_unsigned), .psum_out(sa_psum_unsigned),
        .valid_out(sa_valid_out), .busy(sa_busy)
    );

    // ---------------------------------------------------------
    // 2. PATH 1: Swish Activation (Triggered by valid_in1)
    // ---------------------------------------------------------
    logic signed [SWISH_W-1:0] vpu_swish_out_final [0:TOTAL_W-1];
    logic                      vpu_swish_valid;

    swish_array #(
        .N(TOTAL_W), 
        .WIDTH(DATA_W_OUT), // Processes 32-bit directly
        .FRACT_BITS(FRACT_BITS) 
    ) u_vpu_swish (
        .clk(clk), .rst_n(rst_n),
        .valid_in(sa_valid_out && valid_in1), // Only active if Path 1 was selected
        .x(sa_psum),
        .y(vpu_swish_out_final),
        .valid_out(vpu_swish_valid)
    );

    // ---------------------------------------------------------
    // 3. PATH 2: Layer Normalization (Triggered by valid_in2)
    // ---------------------------------------------------------
    logic signed [DATA_W_OUT-1:0] normalized_output    [0:TOTAL_W-1];
    logic                         norm_final_out_valid;

    layer_normalization_top #(
        .DATA_WIDTH(DATA_W_OUT),
        .K_WIDTH(6),
        .M_WIDTH(17),
        .EMBED_DIM(TOTAL_W) 
    ) u_layernorm (
        .clk(clk),
        .rst_n(rst_n),
        .sum_en_1(sa_valid_out && valid_in2), // Only active if Path 2 was selected
        .activation_in(sa_psum),
        .normalized_output(normalized_output),
        .norm_final_out_valid(norm_final_out_valid)
    );

    logic signed [SWISH_W-1:0] layernorm_out_final [0:TOTAL_W-1];
    generate
        for (c = 0; c < TOTAL_W; c++) begin : NORM_EXPAND
            assign layernorm_out_final[c] = SWISH_W'($signed(normalized_output[c]));
        end
    endgenerate

    // ---------------------------------------------------------
    // 4. PATH 3: Residual Adder
    // ---------------------------------------------------------
    logic signed [SWISH_W-1:0] add_result [0:TOTAL_W-1];
    logic                      add_v1, add_v2, add_v3;

    generate
        for (c = 0; c < TOTAL_W; c++) begin : VPU_ADDER
            assign add_result[c] = skip_conn_in[c] + add_main_in[c];
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            {add_v1, add_v2, add_v3} <= 3'b000;
        end else begin
            add_v1 <= vpu_add_en;
            add_v2 <= add_v1;
            add_v3 <= add_v2;
        end
    end

    // ---------------------------------------------------------
    // 5. The Mega Mux (Final Output Routing)
    // ---------------------------------------------------------
    generate
        for (c = 0; c < TOTAL_W; c++) begin : FINAL_MUX
            always_comb begin
                // Selects the output based on whichever block finishes and asserts valid
                if (norm_final_out_valid)    final_out[c] = layernorm_out_final[c];
                else if (vpu_swish_valid)    final_out[c] = vpu_swish_out_final[c];
                else if (add_v3)             final_out[c] = add_result[c];
                else if (sa_valid_out)       final_out[c] = sa_psum_unsigned[c]; 
                else                         final_out[c] = 0; 
            end
        end
    endgenerate

    assign final_valid_out = norm_final_out_valid | vpu_swish_valid | add_v3 |sa_valid_out;

endmodule