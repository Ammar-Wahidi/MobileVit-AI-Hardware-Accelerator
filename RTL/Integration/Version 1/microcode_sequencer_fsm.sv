module microcode_sequencer_fsm (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start_inference,

    // Status Signals FROM Datapath
    input  logic        sa_done,          // Systolic Array finished its part
    input  logic        final_valid_out,  // Delayed valid from VPU (SA + Requant + Swish)
    
    // Instruction ROM Interface
    output logic [9:0]  instr_mem_addr,
    input  logic [63:0] instr_mem_data,
    
    // Control Signals TO Datapath
    output logic        load_w,       
    output logic        vpu_add_en, 
    output logic        sa_valid_in, 
    output logic        sa_valid_in1,  //sys swish
    output logic        sa_valid_in2,  // sys layer
    
    // Memory Pointers TO SRAM Controller
    output logic [15:0] sram_read_addr_1, 
    output logic [15:0] sram_read_addr_2, 
    output logic [15:0] sram_write_addr,
    output logic [15:0] weight_base_addr, 
    output logic        sram_read_en,
    output logic        sram_write_en,
    
    output logic        inference_done
);

    // =========================================================
    // Internal Registers: PC, IR, and State
    // =========================================================
    logic [9:0]  pc;
    logic [63:0] ir; 
    
    typedef enum logic [3:0] {
        STATE_IDLE       = 4'd0, 
        STATE_FETCH      = 4'd1, 
        STATE_DECODE     = 4'd2, 
        STATE_EXEC_VPU1  = 4'd3, // Combined (Sys + Swish)
        STATE_EXEC_VPU2  = 4'd4, // Combined (Sys + Layer norm)
        STATE_EXEC_VPU3  = 4'd5, // Combined (Sys + Softmax)
        STATE_SYSTOLIC   = 4'd6, // SYSTOLIC
        STATE_EXEC_LOOP  = 4'd7,
        STATE_EXEC_ADD   = 4'd8,
        STATE_WAIT_PIPE  = 4'd9
    } state_t;

    state_t current_state, next_state;

    logic [1:0] loop_counter;
    logic       inc_loop, clr_loop;
    logic       inc_pc, clr_pc, latch_ir;

    // Decoding Logic
    logic [3:0]  op_code;
    logic [15:0] op_weight, op_read1, op_read2, op_write;
    
    assign op_code   = ir[63:60];
    assign op_weight = ir[59:44];
    assign op_read1  = ir[43:28];
    assign op_read2  = ir[27:12];
    assign op_write  = {4'b0, ir[11:0]}; 

    // Sequential Logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= STATE_IDLE;
            loop_counter  <= 2'd0;
            pc            <= 10'd0;
            ir            <= 64'd0;
        end else begin
            current_state <= next_state;
            if (latch_ir) ir <= instr_mem_data; 
            if (clr_loop) loop_counter <= 2'd0;
            else if (inc_loop) loop_counter <= loop_counter + 1;
            if (clr_pc) pc <= 10'd0;
            else if (inc_pc) pc <= pc + 1;
        end
    end

    // Combinational Logic
    always_comb begin
        // Default Assignments
        next_state       = current_state;
        load_w           = 1'b0;
        vpu_add_en       = 1'b0;
        sa_valid_in     = 1'b0;
        sa_valid_in1     = 1'b0;
        sa_valid_in2     = 1'b0;
        sram_read_en     = 1'b0;
        sram_write_en    = final_valid_out; // Write to SRAM when data emerges from Swish
        
        sram_read_addr_1 = op_read1;
        sram_read_addr_2 = op_read2;
        sram_write_addr  = op_write;
        weight_base_addr = op_weight;
        instr_mem_addr   = pc;
        inference_done   = 1'b0;
        inc_loop         = 1'b0; clr_loop = 1'b0;
        inc_pc           = 1'b0; clr_pc = 1'b0;
        latch_ir         = 1'b0;

        case (current_state)
            STATE_IDLE: begin
                clr_loop = 1'b1; clr_pc = 1'b1;
                if (start_inference) next_state = STATE_FETCH;
            end

            STATE_FETCH: begin
                latch_ir   = 1'b1;
                next_state = STATE_DECODE;
            end

            STATE_DECODE: begin
                case (op_code)
                    4'h0: next_state = STATE_EXEC_VPU1;     // sys+swish
                    4'h1: next_state = STATE_EXEC_LOOP;    // 3*(sys+swish)
                    4'h2: next_state = STATE_EXEC_ADD;    //add
                    4'h3: next_state = STATE_EXEC_VPU2;  // sys+layer 
                    4'h4: next_state = STATE_SYSTOLIC;  // sys 
                    4'hF: begin
                        inference_done = 1'b1;
                        next_state     = STATE_IDLE;
                    end
                    default: next_state = STATE_IDLE;
                endcase
            end

            // -------------------------------------------------------------
            // EXECUTION: (Systolic Array -> Swish)
            // -------------------------------------------------------------
            STATE_EXEC_VPU1: begin
                sram_read_en = 1'b1;
                sa_valid_in1  = 1'b1; // Trigger the entire chain
                next_state   = STATE_WAIT_PIPE;
            end
            STATE_EXEC_VPU2: begin
                sram_read_en = 1'b1;
                sa_valid_in2  = 1'b1; // Trigger the entire chain
                next_state   = STATE_WAIT_PIPE;
            end
            STATE_SYSTOLIC: 
            begin
                sram_read_en = 1'b1;
                sa_valid_in  = 1'b1; // Trigger the entire chain
                next_state   = STATE_WAIT_PIPE;
            end

            STATE_EXEC_LOOP: begin
                sram_read_en = 1'b1;
                sa_valid_in1  = 1'b1;
                
                // Ping-Pong Logic (Handles temporary buffers)
                if (loop_counter == 2'd0) begin
                    sram_read_addr_1 = op_read1; sram_write_addr = op_write; 
                end else if (loop_counter == 2'd1) begin
                    sram_read_addr_1 = op_write; sram_write_addr = op_read2; 
                end else begin
                    sram_read_addr_1 = op_read2; sram_write_addr = op_write; 
                end
                next_state = STATE_WAIT_PIPE;
            end

            STATE_EXEC_ADD: begin
                vpu_add_en   = 1'b1;
                sram_read_en = 1'b1;
                sa_valid_in1  = 1'b1;
                next_state   = STATE_WAIT_PIPE;
            end

            // -------------------------------------------------------------
            // WAIT_PIPE: Handling the total latency of the VPU chain
            // -------------------------------------------------------------
            STATE_WAIT_PIPE: begin
                // Hold signals high to keep pipeline filled until done
                if (op_code == 4'h0 || op_code == 4'h1) 
                begin 
                    sram_read_en = 1'b1; 
                    sa_valid_in1 = 1'b1; 
                end
                else if (op_code == 4'h3) 
                begin 
                    sram_read_en = 1'b1; 
                    sa_valid_in2 = 1'b1; 
                end
                else if (op_code == 4'h4) 
                begin 
                    sram_read_en = 1'b1; 
                    sa_valid_in = 1'b1; 
                end
                else if (op_code == 4'h2) 
                begin 
                    sram_read_en = 1'b1; 
                    vpu_add_en = 1'b1; 
                end

                if (final_valid_out) 
                begin
                    if (op_code == 4'h1 && loop_counter < 2'd2) 
                    begin 
                        inc_loop = 1'b1; next_state = STATE_EXEC_LOOP; 
                    end else 
                    begin 
                        clr_loop = 1'b1; inc_pc = 1'b1; next_state = STATE_FETCH; 
                    end
                end
            end
            
            default: next_state = STATE_IDLE;
        endcase
    end
endmodule