module mobilevit_accelerator_top #(
    parameter DATA_W     = 8,
    parameter SWISH_W    = 32,
    parameter N_TILE     = 16,
    parameter TOTAL_W    = 4 * N_TILE // 64
)(
    input  logic                   clk,
    input  logic                   rst_n,
    input  logic                   start_inference,
    
    // --- External SRAM Interface ---
    output logic [15:0]           sram_read_addr_1, 
    output logic [15:0]           sram_read_addr_2, 
    output logic [15:0]           sram_write_addr,
    output logic [15:0]           weight_base_addr, 
    output logic                  sram_read_en,
    output logic                  sram_write_en,
    
    // Input data buses from external SRAM
    input  logic signed [DATA_W-1:0]     sram_act_data    [0:TOTAL_W-1], 
    input  logic signed [DATA_W-1:0]     sram_weight_data [0:TOTAL_W-1], 
    input  logic signed [SWISH_W-1:0]    sram_add_main    [0:TOTAL_W-1], 
    input  logic signed [SWISH_W-1:0]    sram_add_skip    [0:TOTAL_W-1], 
    
    // Output data bus to external SRAM
    output logic signed [SWISH_W-1:0]    sram_write_data  [0:TOTAL_W-1],
    
    // Top-level status
    output logic                  inference_done
);

    // =========================================================
    // Internal Wires
    // =========================================================
    logic [9:0]  pc_addr;
    logic [63:0] current_instruction;
    
    logic        vpu_add_en;
    logic        trigger_datapath; 
    logic        trigger_datapath1; 
    logic        trigger_datapath2; 
    logic        trigger_datapath3; 
    logic        sa_valid_out;
    logic        final_valid_out;

    // =========================================================
    // 1. INSTRUCTION ROM (The Software)
    // =========================================================
    instruction_rom u_rom (
        .pc_addr(pc_addr),
        .instruction(current_instruction)
    );

    // =========================================================
    // 2. MICROCODE SEQUENCER FSM (The Control Unit)
    // =========================================================
    microcode_sequencer_fsm u_controller (
        .clk(clk),
        .rst_n(rst_n),
        .start_inference(start_inference),
        
        // ROM connections
        .instr_mem_addr(pc_addr),
        .instr_mem_data(current_instruction),
        
        // Status from Datapath
        .sa_done(sa_valid_out),
        .final_valid_out(final_valid_out),
        
        // Control to Datapath
        .vpu_add_en(vpu_add_en),
        .sa_valid_in(trigger_datapath), 
        .sa_valid_in1(trigger_datapath1), 
        .sa_valid_in2(trigger_datapath2), 
        .sa_valid_in3(trigger_datapath3), 
        
        // Control to External SRAM
        .sram_read_addr_1(sram_read_addr_1),
        .sram_read_addr_2(sram_read_addr_2),
        .sram_write_addr(sram_write_addr),
        .weight_base_addr(weight_base_addr),
        .sram_read_en(sram_read_en),
        .sram_write_en(sram_write_en),
        
        .inference_done(inference_done)
    );

    // =========================================================
    // 3. LEGO DATAPATH (The Computation Engine)
    // =========================================================
    mobilevit_lego_datapath #(
        .DATA_W(DATA_W),
        .DATA_W_OUT(SWISH_W), 
        .SWISH_W(SWISH_W),
        .FRACT_BITS(1),
        .N_TILE(N_TILE)
    ) u_datapath (
        .clk(clk),
        .rst_n(rst_n),
        
        // Control
        .valid_in(trigger_datapath),
        .valid_in1(trigger_datapath1), 
        .valid_in2(trigger_datapath2), 
        .valid_in3(trigger_datapath3), 
        
        .lego_type(2'd1),      
        .y_input_size(8'd16), 
        .transpose_en(1'b0),
        .vpu_add_en(vpu_add_en),
        
        // Data in
        .act_in(sram_act_data),
        .weight_in(sram_weight_data),
        .skip_conn_in(sram_add_skip), 
        .add_main_in(sram_add_main),  
        
        // Data out
        .final_out(sram_write_data),
        .final_valid_out(final_valid_out),
        .sa_valid_out(sa_valid_out),
        .sa_busy() 
    );

endmodule