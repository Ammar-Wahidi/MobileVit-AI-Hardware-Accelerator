#include <iostream>
#include <vector>
#include <cstdint>
#include <string>

// ============================================================
// Constants & Structures
// ============================================================
static constexpr uint32_t DEFAULT_NULL_ADDR = 999;

struct MemoryAccess {
    uint32_t word_addr;
    uint8_t  byte_sel;
    uint32_t bank_id;
};

struct HardwareTile {
    std::vector<MemoryAccess> A;
    std::vector<MemoryAccess> B;
    std::vector<MemoryAccess> C;

    bool final_acc;     // this C tile is complete
    bool layer_done;    // whole matrix done
    bool busy;
};

// ============================================================
// SRAM address packing (byte-addressable)
// ============================================================
inline MemoryAccess pack_sram_addr(uint32_t elem_index,
                                   uint32_t bank_id,
                                   uint32_t NULL_ADDR = DEFAULT_NULL_ADDR)
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
// Debug Print
// ============================================================
void print_tiles(const std::vector<HardwareTile>& tiles,
                 const std::string& tag)
{
    std::cout << "\n========== " << tag << " ==========\n";

    for (size_t t = 0; t < tiles.size(); t++) {
        const auto& tile = tiles[t];

        std::cout << "TILE " << t
                  << " | final_acc=" << tile.final_acc
                  << " | layer_done=" << tile.layer_done
                  << "\n";

        auto dump = [](const std::vector<MemoryAccess>& v) {
            for (auto& m : v) {
                std::cout << "[W:" << m.word_addr
                          << " B:" << int(m.byte_sel)
                          << " BID:" << m.bank_id << "] ";
            }
            std::cout << "\n";
        };

        std::cout << " A: "; dump(tile.A);
        std::cout << " B: "; dump(tile.B);
        std::cout << " C: "; dump(tile.C);
        std::cout << "---------------------------------\n";
    }
}

// ============================================================
// TEST
// ============================================================
int main()
{
    uint32_t base_a = 0;
    uint32_t base_b = 0;
    uint32_t base_c = 0;

    uint32_t F_M = 16;
    uint32_t F_N = 16;
    uint32_t F_K = 16;

    uint32_t TM = 8;
    uint32_t TN = 8;
    uint32_t TK = 8;

    std::vector<HardwareTile> tiles;

    // 2 x 2 x 2 = 8 tile steps
    for (int i = 0; i < 8; i++) {
        matmul_agu(
            base_a, base_b, base_c, DEFAULT_NULL_ADDR,
            F_M, F_N, F_K,
            false,
            TM, TN, TK,
            tiles
        );
    }

    print_tiles(tiles, "FINAL — COLUMN-STREAMING B");

    return 0;
}
