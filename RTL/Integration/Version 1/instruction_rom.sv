module instruction_rom (
    input  logic [9:0]  pc_addr,      // Program Counter from the FSM
    output logic [63:0] instruction   // The 64-bit machine code
);

    // 64-bit Instruction Format:
    // [63:60] Opcode (4 bits = 1 Hex digit)
    // [59:44] Weight Base Address (16 bits = 4 Hex digits)
    // [43:28] Read Address 1 (16 bits = 4 Hex digits)
    // [27:12] Read Address 2 (16 bits = 4 Hex digits)
    // [11:0]  Write Address (12 bits = 3 Hex digits)
    // Total Hex Digits = 1 + 4 + 4 + 4 + 3 = 16 Digits (Exactly 64 bits!)

    always_comb begin
    case (pc_addr)
        // 1. (sys+swish) >> Buff A (Address 100)
        10'd0: instruction = 64'h3_0100_0000_0000_100;
        //10'd0: instruction = 64'h1_0101_0100_0300_400; 
/*
        // 2. Buff A >> (sys+swish)*3 >> Buff D (Address 400)
        // Reads from 100, uses 300 as temp workspace, final result in 400
        10'd1: instruction = 64'h1_0101_0100_0300_400; 

        // 3. Buff A + Buff D >> x (Address 500)
        10'd2: instruction = 64'h2_0000_0100_0400_500; 

        // 4. x >> (sys+swish)*3 >> y (Address 600)
        10'd3: instruction = 64'h1_0102_0500_0300_600; 

        // 5. y >> (sys+swish)*3 >> e (Address 700)
        10'd4: instruction = 64'h1_0103_0600_0300_700; 

        // 6. y + e >> h (Address 800)
        10'd5: instruction = 64'h2_0000_0600_0700_800; 

        // 7. h >> (sys+swish)*3 >> i (Address 900)
        10'd6: instruction = 64'h1_0104_0800_0300_900; 

        // 8. h + i >> z (Address A00)
        10'd7: instruction = 64'h2_0000_0800_0900_A00; 

        // 9. z >> (sys+swish)*3 >> final (Address B00)
        10'd8: instruction = 64'h1_0105_0A00_0300_B00; 
*/
        // 10. HALT
        10'd9: instruction = 64'hF_0000_0000_0000_000; 
        10'd10: instruction = 64'hF_0000_0000_0000_000; 
        10'd11: instruction = 64'hF_0000_0000_0000_000; 

        default: instruction = 64'hF_0000_0000_0000_000;
    endcase
end

endmodule