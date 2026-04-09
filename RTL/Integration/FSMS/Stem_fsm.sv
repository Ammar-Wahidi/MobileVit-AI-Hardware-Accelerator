module stem_fsm (
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
    output logic stem_done
);



typedef enum logic [2:0] {
        STATE_IDLE       = 3'd0,
        STATE_SYS        = 3'd1,
        STATE_SCALE      = 3'd2,
        STATE_SWISH      = 3'd3,
        STATE_REQ        = 3'd4,
        STATE_MEMORY     = 3'd5,
        STATE_DONE_STEM  = 3'd6
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
        stem_done    = 1'b0;

        case (current_state)
        STATE_IDLE:
        begin
            if(start)
            next_state = STATE_SYS;
            else
            next_state = STATE_IDLE;
        end 
        STATE_SYS:
        begin
        sys_en = 1'b1;
            if(sys_done)
            next_state = STATE_SCALE;
            else
            next_state = STATE_SYS;
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
            next_state = STATE_REQ;
            else
            next_state = STATE_SWISH;
        end 
        STATE_REQ:
        begin
        req_en = 1'b1;
            if(req_done)
            next_state = STATE_MEMORY;
            else
            next_state = STATE_REQ;
        end 
        STATE_MEMORY:
        begin
        mem_write_en = 1'b1;
            if(mem_done)
            next_state = STATE_DONE_STEM;
            else
            next_state = STATE_MEMORY;
        end  
        STATE_DONE_STEM:
        begin
            stem_done = 1'b1;
            next_state = STATE_IDLE;
        end 
         
        default:next_state = STATE_IDLE; 
        endcase
    end
endmodule