#include <vector> 
#include <cstdint> 
#include <algorithm> 
#include <iostream> 
#include <string> 
#include <iomanip>

// ============================================================
// Structures
// ============================================================
struct MemoryAccess {
    uint32_t word_addr;
    uint8_t  byte_sel;
    uint32_t bank_id;
};

struct HardwareTile {
    std::vector<MemoryAccess> A;
    std::vector<MemoryAccess> B;
    std::vector<MemoryAccess> C;
    bool final_acc;
    bool layer_done;    // whole matrix done
    bool busy;
};


// ============================================================
// SRAM packing
// ============================================================
inline MemoryAccess pack_sram_addr(uint32_t elem_index,
                                   uint32_t bank_id,
                                   uint32_t NULL_ADDR)
{
    if (elem_index == NULL_ADDR)
        return { NULL_ADDR >> 2, 0, bank_id };

    return {
        elem_index >> 2,
        uint8_t(elem_index & 0x3),
        bank_id
    };
}



// ============================================================
// CONVOLUTION AGU (CORRECT + REFERENCE-EQUIVALENT)
// ============================================================
void conv_agu(
    /* base addresses (element indexed) */
    uint32_t base_a,
    uint32_t base_b,
    uint32_t NULL_ADDR,
    uint32_t base_c,

    /* input feature map config */
    uint32_t F_Cout,
    uint32_t ch_Cin,
    uint32_t ch_W,
    uint32_t ch_H,

    /* kernel */
    uint32_t KW,
    uint32_t KH,

    /* control */
    bool     last_C,   // last chunk in this layer
    uint32_t P,        // padding
    uint32_t S,        // stride
    uint32_t mode,     // 0=Regular, 1=Pointwise, 2=Depthwise

    /* tile sizes */
    uint32_t TM,
    uint32_t TN,
    uint32_t TK,

    /* output */
    std::vector<HardwareTile>& tiles
) {
    // ========================================================
    // Persistent C stacking (matches reference AGU)
    // ========================================================
    static uint32_t global_c_offset = 0;

    // ========================================================
    // Effective kernel
    // ========================================================
    uint32_t Kh_e = (mode == 1) ? 1 : KH;
    uint32_t Kw_e = (mode == 1) ? 1 : KW;

    // ========================================================
    // Output feature map size
    // ========================================================
    uint32_t Ho = (ch_H + 2 * P - Kh_e) / S + 1;
    uint32_t Wo = (ch_W + 2 * P - Kw_e) / S + 1;

    uint32_t M_c = Ho * Wo;
    uint32_t N_c = (mode == 2) ? ch_Cin : F_Cout;
    uint32_t K_block = KH * KW;

    uint32_t K_c;
    if (mode == 0)      K_c = KH * KW * ch_Cin;  // regular
    else if (mode == 1) K_c = ch_Cin;            // pointwise
    else                K_c = K_block * ch_Cin; // depthwise

    if (TK == 0) TK = K_c;

    // ========================================================
    // TILING LOOPS
    // ========================================================
    for (uint32_t i_t = 0; i_t < M_c; i_t += TM)
    for (uint32_t j_t = 0; j_t < N_c; j_t += TN)
    for (uint32_t k_t = 0; k_t < K_c; k_t += TK) {

        HardwareTile ht;
        ht.busy     = true;
        ht.final_acc = (k_t + TK >= K_c);  // K-loop final only

        uint32_t eTM = std::min(TM, M_c - i_t);
        uint32_t eTN = std::min(TN, N_c - j_t);
        uint32_t eTK = std::min(TK, K_c - k_t);

        // ====================================================
        // PORT A — INPUT FEATURE MAP
        // ====================================================
        for (uint32_t i = 0; i < TM; i++)
        for (uint32_t k = 0; k < TK; k++) {

            uint32_t adr = NULL_ADDR;

            if (i < eTM && k < eTK) {
                uint32_t ig = i_t + i;
                uint32_t kg = k_t + k;

                uint32_t oh = ig / Wo;
                uint32_t ow = ig % Wo;

                int ih, iw;
                uint32_t c;

                if (mode == 2) {
                    // Depthwise
                    c  = kg / K_block;
                    ih = oh * S - P + (kg % K_block) / KW;
                    iw = ow * S - P + (kg % K_block) % KW;
                } else {
                    uint32_t kh_i = (mode == 1) ? 0 : kg / (KW * ch_Cin);
                    uint32_t kw_i = (mode == 1) ? 0 : (kg % (KW * ch_Cin)) / ch_Cin;
                    c  = (mode == 1) ? kg : kg % ch_Cin;
                    ih = oh * S - P + kh_i;
                    iw = ow * S - P + kw_i;
                }

                if (ih >= 0 && ih < (int)ch_H &&
                    iw >= 0 && iw < (int)ch_W)
                {
                    adr = base_a + (ih * ch_W + iw) * ch_Cin + c;
                }
            }

            ht.A.push_back(
                pack_sram_addr(adr, k, NULL_ADDR)
            );
        }

        // ====================================================
        // PORT B — WEIGHTS
        // ====================================================
        for (uint32_t j = 0; j < TN; j++)
        for (uint32_t k = 0; k < TK; k++) {

            uint32_t adr = NULL_ADDR;

            if (j < eTN && k < eTK) {
                uint32_t jg = j_t + j;
                uint32_t kg = k_t + k;

                if (mode == 2) {
                    // Depthwise
                    if (kg >= jg * K_block &&
                        kg < (jg + 1) * K_block)
                        adr = base_b + kg;
                } else {
                    adr = base_b + jg * K_c + kg;
                }
            }

            ht.B.push_back(
                pack_sram_addr(adr, k, NULL_ADDR)
            );
        }

        // ====================================================
        // PORT C — OUTPUT (REFERENCE-EQUIVALENT)
        // ====================================================
        for (uint32_t i = 0; i < TM; i++)
        for (uint32_t j = 0; j < TN; j++) {

            uint32_t adr = NULL_ADDR;

            if (ht.final_acc && i < eTM && j < eTN) {
                adr = base_c + global_c_offset +
                      (i_t + i) * N_c + (j_t + j);
            }

            ht.C.push_back(
                pack_sram_addr(adr, 0, NULL_ADDR)
            );
        }

        tiles.push_back(ht);
    }

    // ========================================================
    // Update global C offset (CRITICAL)
    // ========================================================
    if (last_C) {
        global_c_offset = 0;               // new layer
    } else {
        global_c_offset += M_c * N_c;      // stack outputs
    }

    if (!tiles.empty())
        tiles.back().busy = false;
}

// ============================================================
// MATRIX-MUL AGU (FINAL VERSION)
//   - A : tile-local, row-major
//   - B : tile-local, COLUMN streaming
//   - C : global matrix
// ============================================================
void matmul_agu(
    uint32_t base_a,
    uint32_t base_b,
    uint32_t base_c,
    uint32_t NULL_ADDR,

    uint32_t F_M,
    uint32_t F_N,
    uint32_t F_K,

    bool last_C,

    uint32_t TM,
    uint32_t TN,
    uint32_t TK,

    std::vector<HardwareTile>& out_tiles
) {
    // --------------------------------------------------------
    // Persistent traversal state
    // --------------------------------------------------------
    static uint32_t tile_row = 0;
    static uint32_t tile_col = 0;
    static uint32_t tile_k   = 0;

    const uint32_t NUM_TILE_M = F_M / TM;
    const uint32_t NUM_TILE_N = F_N / TN;
    const uint32_t NUM_TILE_K = F_K / TK;

    HardwareTile ht{};
    ht.busy = true;

    // --------------------------------------------------------
    // FINAL_ACC & LAYER_DONE
    // --------------------------------------------------------
    ht.final_acc = (tile_k == NUM_TILE_K - 1);

    ht.layer_done =
        ht.final_acc &&
        (tile_row == NUM_TILE_M - 1) &&
        (tile_col == NUM_TILE_N - 1);

    // --------------------------------------------------------
    // Port A  [TM x TK]  (ROW-MAJOR)
    // --------------------------------------------------------
    for (uint32_t i = 0; i < TM; i++)
        for (uint32_t k = 0; k < TK; k++) {
            uint32_t a_idx = base_a + i * TK + k;
            ht.A.push_back(pack_sram_addr(a_idx, k, NULL_ADDR));
        }

    // --------------------------------------------------------
    // Port B  [TK x TN]  (COLUMN STREAMING)  <<< FIXED
    // --------------------------------------------------------
    for (uint32_t j = 0; j < TN; j++)        // column first
        for (uint32_t k = 0; k < TK; k++) {  // then rows
            uint32_t b_idx = base_b + k * TN + j;
            ht.B.push_back(pack_sram_addr(b_idx, j, NULL_ADDR));
        }

    // --------------------------------------------------------
    // Port C  [TM x TN]  (GLOBAL MATRIX)
    // --------------------------------------------------------
    for (uint32_t i = 0; i < TM; i++)
        for (uint32_t j = 0; j < TN; j++) {

            uint32_t c_row = tile_row * TM + i;
            uint32_t c_col = tile_col * TN + j;

            uint32_t c_idx =
                base_c +
                c_row * F_N +
                c_col;

            ht.C.push_back(pack_sram_addr(c_idx, 0, NULL_ADDR));
        }

    out_tiles.push_back(ht);
    ht.busy = false;

    // --------------------------------------------------------
    // Advance traversal
    // --------------------------------------------------------
    if (ht.final_acc) {
        tile_k = 0;

        tile_col++;
        if (tile_col == NUM_TILE_N) {
            tile_col = 0;
            tile_row++;
        }
    } else {
        tile_k++;
    }

    // --------------------------------------------------------
    // Reset for next layer
    // --------------------------------------------------------
    if (ht.layer_done || last_C) {
        tile_row = tile_col = tile_k = 0;
    }
}


// ============================================================
// TOP LEVEL AGU
// ============================================================
std::vector<HardwareTile> agu(const uint32_t regs[7]) // Assuming 7 registers (0-6)
{
    std::vector<HardwareTile> tiles;

    // 1. Control & Common Config
    bool start    = (regs[6] >> 31) & 0x1;
    if (!start) return tiles;

    uint32_t base_a   = regs[0];
    uint32_t base_b   = regs[1];
    uint32_t NULL_ADR = regs[2];
    uint32_t base_c   = regs[3];

    // 2. Mode & Tile Sizes (Reg6)
    // Mask mode to 2 bits [29:28] to avoid overlap with last_C [30]
    uint32_t mode   = (regs[6] >> 28) & 0x3; 
    bool last_C     = (regs[6] >> 30) & 0x1;

    uint32_t TM = (regs[6] >> 16) & 0xFF;
    uint32_t TN = (regs[6] >> 8)  & 0xFF;
    uint32_t TK = (regs[6] >> 0)  & 0xFF;

    // 3. MATMUL BRANCH (Mode 3)
    if (mode == 3) {
        // Correct indices for F_K, F_N, F_M from image_1217a7
        uint32_t F_K = (regs[5] >> 18) & 0x1FF;
        uint32_t F_N = (regs[5] >> 9)  & 0x1FF;
        uint32_t F_M = (regs[5] >> 0)  & 0x1FF;

        matmul_agu(base_a, base_b, base_c, NULL_ADR,
                   F_M, F_N, F_K, last_C, TM, TN, TK, tiles);
        return tiles;
    }

    // 4. CONVOLUTION BRANCH (Mode 0, 1, 2)
    uint32_t F_Cout = (regs[4] >> 24) & 0xFF;
    uint32_t ch_Cin = (regs[4] >> 16) & 0xFF;
    uint32_t ch_W   = (regs[4] >> 8)  & 0xFF;
    uint32_t ch_H   = (regs[4] >> 0)  & 0xFF;

    // Correct Kernel indices for KH/KW
    uint32_t KW = (regs[5] >> 29) & 0x3;
    uint32_t KH = (regs[5] >> 27) & 0x3;

    uint32_t P  = (regs[6] >> 26) & 0x3;
    uint32_t S  = (regs[6] >> 24) & 0x3;

    conv_agu(base_a, base_b, NULL_ADR, base_c,
             F_Cout, ch_Cin, ch_W, ch_H, KW, KH,
             last_C, P, S, mode, TM, TN, TK, tiles);

    return tiles;
}





//========================================================================
//  ----- TEST -------------------
//========================================================================
void print_tiles(const std::vector<HardwareTile>& tiles,
                 const std::string& tag)
{
    std::cout << "\n========== " << tag << " ==========\n";

    for (size_t t = 0; t < tiles.size(); t++) {
        const auto& tile = tiles[t];

        std::cout << "TILE " << t
                  << " | BUSY=" << tile.busy
                  << " | FINAL=" << tile.final_acc
                  << "\n";

        auto dump = [](const std::vector<MemoryAccess>& v) {
            for (const auto& m : v) {
                std::cout << "[W:" << m.word_addr
                          << " B:" << int(m.byte_sel)
                          << " bid:" << m.bank_id
                          << "] ";
            }
            std::cout << "\n";
        };

        std::cout << "A: ";
        dump(tile.A);

        std::cout << "B: ";
        dump(tile.B);

        std::cout << "C: ";
        dump(tile.C);

        std::cout << "--------------------------------------\n";
    }
}

int main()
{
    // --------------------------------------------------
    // Test Variables (Modify these to change test case)
    // --------------------------------------------------
    uint32_t addr_a = 0x0000;   // Base A
    uint32_t addr_b = 0x1000;   // Base B
    uint32_t addr_c = 0x8000;   // Base C (Global)
    uint32_t null_a = 0xFFFF;   // Null Address

    // Full Matrix Dimensions
    uint32_t fm = 4;            // F_M
    uint32_t fn = 4;            // F_N
    uint32_t fk = 4;            // F_K

    // Tile Dimensions
    uint32_t tm = 2;            
    uint32_t tn = 2;            
    uint32_t tk = 2;            

    // Convolution/IFM Params (Used if mode < 3)
    uint32_t f_cout = 16;
    uint32_t ch_cin = 8;
    uint32_t ch_w   = 32;
    uint32_t ch_h   = 32;
    uint32_t kw     = 3;
    uint32_t kh     = 3;
    uint32_t pad    = 1;
    uint32_t stride = 1;

    // Control
    uint32_t mode   = 3;        // 3 = MATMUL
    bool start      = true;
    bool last_c     = true;

    // --------------------------------------------------
    // HW-style Register File Assembly
    // --------------------------------------------------
    uint32_t regs[7] = {0};

    // Addresses
    regs[0] = addr_a;   // Reg0: Base A [31:0]
    regs[1] = addr_b;   // Reg1: Base B [31:0]
    regs[2] = null_a;   // Reg2: Null [31:0]
    regs[3] = addr_c;   // Reg3: Base C [31:0]

    // IFM Config
    regs[4] = (f_cout << 24) | (ch_cin << 16) | (ch_w << 8) | (ch_h << 0);

    // Full Matrix & Kernel
    regs[5] = ((kw & 0x3) << 29) | 
              ((kh & 0x3) << 27) | 
              ((fk & 0x1FF) << 18)| 
              ((fn & 0x1FF) << 9) | 
              ((fm & 0x1FF) << 0);

    // Tiles Config & Control
    regs[6] = (uint32_t(start) << 31)  | 
              (uint32_t(last_c) << 30) | 
              ((mode & 0x3) << 28)     | 
              ((pad & 0x3) << 26)      | 
              ((stride & 0x3) << 24)   |
              ((tm & 0xFF) << 16)      | 
              ((tn & 0xFF) << 8)       | 
              ((tk & 0xFF) << 0);


    // Run AGU and Print results
    std::vector<HardwareTile> all_tiles;
    
    // For a 4x4x4 matrix with 2x2x2 tiles, we expect 8 tiles total ( (4/2)^3 )
    int expected_tiles = (fm/tm) * (fn/tn) * (fk/tk);
    
    std::cout << "Running AGU for " << expected_tiles << " steps...\n";
    
    for(int i = 0; i < expected_tiles; ++i) {
        auto step_tiles = agu(regs);
        if(!step_tiles.empty()) {
            all_tiles.push_back(step_tiles[0]);
        }
    }

    print_tiles(all_tiles, "MATMUL 4x4x4 TEST");

    return 0;
}