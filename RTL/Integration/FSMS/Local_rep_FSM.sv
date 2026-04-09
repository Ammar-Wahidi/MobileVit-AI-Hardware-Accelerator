module local_rep_fsm (
    input  logic clk,
    input  logic rst,
    input  logic start,

    input  logic sys_done,
    input  logic sca_done,
    input  logic swi_done,
    input  logic req_done,
    input  logic mem_done,
    output logic sys_en,
    output logic scale_en,
    output logic swish_en,
    output logic req_en,
    output logic mem_write_en,
    output logic local_done
);



typedef enum logic [3:0] {
        STATE_IDLE       = 4'd0,
        STATE_SYS_1      = 4'd1,
        STATE_SYS_2      = 4'd2,
        STATE_SCALE      = 4'd3,
        STATE_SWISH      = 4'd4,
        STATE_REQ_1      = 4'd5,
        STATE_REQ_2      = 4'd6,
        STATE_MEMORY_1   = 4'd7,
        STATE_MEMORY_2   = 4'd8,
        STATE_DONE_LOCAL = 4'd9
    } state_t;

    state_t current_state, next_state;

    always@(posedge clk or negedge rst) 
    begin
        if (!rst) 
        begin
            current_state <= STATE_IDLE;
        end else 
        begin
            current_state <= next_state;
        end
    end


    always @(*) 
    begin
        next_state   = current_state;
        sys_en       = 1'b0;
        scale_en     = 1'b0;
        swish_en     = 1'b0;
        req_en       = 1'b0;
        mem_write_en = 1'b0;
        local_done    = 1'b0;

        case (current_state)
        STATE_IDLE:
        begin
            if(start)
            next_state = STATE_SYS_1;
            else
            next_state = STATE_IDLE;
        end 
        STATE_SYS_1:
        begin
        sys_en = 1'b1;
            if(sys_done)
            next_state = STATE_SCALE;
            else
            next_state = STATE_SYS_1;
        end
        STATE_SCALE:
        begin
        scale_en = 1'b1;
            if(sca_done)
            next_state = STATE_SWISH;
            else
            next_state = STATE_SCALE;
        end 
        STATE_SWISH:
        begin
        swish_en = 1'b1;
            if(swi_done)
            next_state = STATE_REQ_1;
            else
            next_state = STATE_SWISH;
        end 
        STATE_REQ_1:
        begin
        req_en = 1'b1;
            if(req_done)
            next_state = STATE_MEMORY_1;
            else
            next_state = STATE_REQ_1;
        end 
        STATE_MEMORY_1:
        begin
        mem_write_en = 1'b1;
            if(mem_done)
            next_state = STATE_SYS_2;
            else
            next_state = STATE_MEMORY_1;
        end
        STATE_SYS_2:
        begin
        sys_en = 1'b1;
            if(sys_done)
            next_state = STATE_REQ_2;
            else
            next_state = STATE_SYS_2;
        end
        STATE_REQ_2:
        begin
        req_en = 1'b1;
            if(req_done)
            next_state = STATE_MEMORY_2;
            else
            next_state = STATE_REQ_2;
        end
        STATE_MEMORY_2:
        begin
        mem_write_en = 1'b1;
            if(mem_done)
            next_state = STATE_DONE_LOCAL;
            else
            next_state = STATE_MEMORY_2;
        end   
        STATE_DONE_LOCAL:
        begin
            local_done = 1'b1;
            next_state = STATE_IDLE;
        end 
         
        default:next_state = STATE_IDLE; 
        endcase
    end
endmodule