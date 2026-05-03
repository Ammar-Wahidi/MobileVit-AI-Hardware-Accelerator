#include <iostream>
#include <iomanip>
#include "hls_stream.h"
#include "ap_int.h"

// Prototype
void agu_top(
    hls::stream<ap_uint<32>>& config_regs,
    ap_uint<32>& addr_a, ap_uint<4>& be_a,
    ap_uint<32>& addr_b, ap_uint<4>& be_b,
    ap_uint<32>& addr_c, ap_uint<4>& be_c,
    bool& busy, bool& final_acc, bool& layer_done
);

int main() {
    hls::stream<ap_uint<32>> config_fifo;

    // --- Configuration: Pointwise Conv (Mode 1) ---
    // Matrix size effectively 4x4x4 (M=16, N=1, K=4)
    uint32_t base_a = 0x1000;
    uint32_t base_b = 0x2000;
    uint32_t null_adr = 0xFFFF;
    uint32_t base_c = 0x3000;

    // Reg 4: Cout=4, Cin=4, W=2, H=2 (F_Cout[31:24], Cin[23:16], W[15:8], H[7:0])
    uint32_t reg4 = (4 << 24) | (4 << 16) | (2 << 8) | (2 << 0);
    // Reg 5: Mode 1 (KH/KW don't matter much here)
    uint32_t reg5 = 0; 
    // Reg 6: Mode=1 (PW), TM=2, TN=2, TK=2, P=0, S=1
    // (Mode[29:28], TM[23:16], TN[15:8], TK[7:0])
    uint32_t reg6 = (1 << 28) | (2 << 16) | (2 << 15) | (2 << 0);

    // Push configuration
    config_fifo.write(base_a);
    config_fifo.write(base_b);
    config_fifo.write(null_adr);
    config_fifo.write(base_c);
    config_fifo.write(reg4);
    config_fifo.write(reg5);
    config_fifo.write(reg6);

    // Outputs
    ap_uint<32> aa, ab, ac;
    ap_uint<4> bea, beb, bec;
    bool bsy, fa, ld;

    std::cout << std::left << std::setw(6) << "Cycle" 
              << std::setw(10) << "A_Addr" << std::setw(5) << "BE"
              << std::setw(10) << "B_Addr" << std::setw(5) << "BE"
              << std::setw(10) << "C_Addr" << "Flags" << std::endl;
    std::cout << std::string(65, '-') << std::endl;

    // Run for enough cycles to see tiling logic
    for (int i = 0; i < 50; i++) {
        agu_top(config_fifo, aa, bea, ab, beb, ac, bec, bsy, fa, ld);
        
        if (bsy) {
            std::cout << std::dec << std::setw(6) << i 
                      << "0x" << std::hex << std::setw(8) << aa.to_uint() << std::dec << std::setw(5) << (int)bea
                      << "0x" << std::hex << std::setw(8) << ab.to_uint() << std::dec << std::setw(5) << (int)beb
                      << "0x" << std::hex << std::setw(8) << ac.to_uint() 
                      << (fa ? " [F_ACC]" : "") 
                      << (ld ? " [DONE]" : "") << std::endl;
        }
        
        if (ld) break; // Stop when the AGU says the layer is finished
    }

    return 0;
}