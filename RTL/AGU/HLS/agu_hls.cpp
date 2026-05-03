#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

inline ap_uint<4> get_be_final(uint32_t addr) {
    #pragma HLS INLINE
    return (ap_uint<4>)(1 << (addr & 0x3));
}

void agu_top(
    hls::stream<ap_uint<32>>& config_regs,
    ap_uint<32>& addr_a, ap_uint<4>& be_a,
    ap_uint<32>& addr_b, ap_uint<4>& be_b,
    ap_uint<32>& addr_c, ap_uint<4>& be_c,
    bool& busy, bool& final_acc, bool& layer_done
) {
    #pragma HLS INTERFACE axis port=config_regs
    #pragma HLS INTERFACE ap_none port=addr_a, be_a, addr_b, be_b, addr_c, be_c
    #pragma HLS INTERFACE ap_none port=busy, final_acc, layer_done

    // --- Persistence ---
    static bool configured = false;
    static uint8_t config_cnt = 0;
    static uint32_t s_base_a, s_base_b, s_base_c, s_null;
    static uint32_t s_F_Cout, s_Cin, s_W, s_H, s_KH, s_KW;
    static uint32_t s_TM, s_TN, s_TK, s_P, s_S, s_mode;
    static uint32_t s_Ho, s_Wo, s_Mc, s_Nc, s_Kc;
    static uint32_t global_c_offset = 0;

    static uint32_t it=0, jt=0, kt=0; 
    static uint32_t i=0, j=0, k=0;

    // 1. Configuration Fetch (Sequential Read)
    if (!configured) {
        if (!config_regs.empty()) {
            ap_uint<32> val = config_regs.read();
            
            switch(config_cnt) {
                case 0: s_base_a = val; break;
                case 1: s_base_b = val; break;
                case 2: s_null   = val; break;
                case 3: s_base_c = val; break;
                case 4: 
                    s_F_Cout = (val >> 24) & 0xFF;
                    s_Cin    = (val >> 16) & 0xFF;
                    s_W      = (val >> 8)  & 0xFF;
                    s_H      = (val >> 0)  & 0xFF;
                    break;
                case 5:
                    s_KH = (val >> 27) & 0x7;
                    s_KW = (val >> 24) & 0x7;
                    // For MatMul, these bits represent F_K, F_N, F_M
                    s_Kc = (val >> 18) & 0x1FF;
                    s_Nc = (val >> 9)  & 0x1FF;
                    s_Mc = (val >> 0)  & 0x1FF;
                    break;
                case 6:
                    s_mode = (val >> 28) & 0x3;
                    s_TM   = (val >> 16) & 0xFF;
                    s_TN   = (val >> 8)  & 0xFF;
                    s_TK   = (val >> 0)  & 0xFF;
                    s_P    = (val >> 26) & 0x3;
                    s_S    = (val >> 24) & 0x3;
                    
                    bool last_C = (val >> 31) & 0x1;
                    if (last_C) global_c_offset = 0;

                    // Compute derived params
                    if (s_mode != 3) { // Conv Mode
                        uint32_t Kh_e = (s_mode == 1) ? 1 : s_KH;
                        uint32_t Kw_e = (s_mode == 1) ? 1 : s_KW;
                        uint32_t stride = (s_S == 0) ? 1 : s_S;
                        s_Ho = (s_H + 2 * s_P - Kh_e) / stride + 1;
                        s_Wo = (s_W + 2 * s_P - Kw_e) / stride + 1;
                        s_Mc = s_Ho * s_Wo;
                        s_Nc = (s_mode == 2) ? s_Cin : s_F_Cout;
                        s_Kc = (s_mode == 0) ? (s_KH * s_KW * s_Cin) : 
                               (s_mode == 1) ? s_Cin : (s_KH * s_KW * s_Cin);
                    }
                    
                    // Reset counters for new layer
                    it=0; jt=0; kt=0; i=0; j=0; k=0;
                    configured = true;
                    config_cnt = 0;
                    break;
            }
            if (!configured) config_cnt++;
        }
        busy = false;
        layer_done = false;
    }

    // 2. Execution Logic
    if (configured) {
        busy = true;
        uint32_t adr_a = s_null, adr_b = s_null, adr_c = s_null;

        uint32_t ig = it + i;
        uint32_t jg = jt + j;
        uint32_t kg = kt + k;

        // Port A/B Addressing
        if (ig < s_Mc && kg < s_Kc) {
            if (s_mode == 3) {
                adr_a = s_base_a + (i * s_TK) + k;
            } else if (s_Wo != 0) {
                uint32_t oh = ig / s_Wo;
                uint32_t ow = ig % s_Wo;
                int ih = oh * s_S - s_P; 
                int iw = ow * s_S - s_P;
                uint32_t c;

                if (s_mode == 2) { // Depthwise
                    uint32_t K_bl = s_KH * s_KW;
                    c = kg / K_bl;
                    ih += (kg % K_bl) / s_KW;
                    iw += (kg % K_bl) % s_KW;
                } else { // Reg/Pointwise
                    uint32_t K_sub = s_KW * s_Cin;
                    ih += (s_mode == 1) ? 0 : (kg / K_sub);
                    iw += (s_mode == 1) ? 0 : ((kg % K_sub) / s_Cin);
                    c  = (s_mode == 1) ? kg : (kg % s_Cin);
                }

                if (ih >= 0 && ih < (int)s_H && iw >= 0 && iw < (int)s_W)
                    adr_a = s_base_a + (ih * s_W + iw) * s_Cin + c;
            }
        }

        if (jg < s_Nc && kg < s_Kc) {
            if (s_mode == 3) adr_b = s_base_b + (k * s_TN) + j;
            else if (s_mode == 2) {
                uint32_t K_bl = s_KH * s_KW;
                if (kg >= jg * K_bl && kg < (jg + 1) * K_bl) adr_b = s_base_b + kg;
            } else adr_b = s_base_b + jg * s_Kc + kg;
        }

        // Port C Addressing
        final_acc = (kt + s_TK >= s_Kc);
        if (final_acc && ig < s_Mc && jg < s_Nc) {
            adr_c = s_base_c + global_c_offset + ig * s_Nc + jg;
        }

        addr_a = adr_a >> 2; be_a = get_be_final(adr_a);
        addr_b = adr_b >> 2; be_b = get_be_final(adr_b);
        addr_c = adr_c >> 2; be_c = get_be_final(adr_c);

        // 3. Increment Counters
        k++;
        if (k >= s_TK) {
            k = 0; j++;
            if (j >= s_TN) {
                j = 0; i++;
                if (i >= s_TM) {
                    i = 0; kt += s_TK;
                    if (kt >= s_Kc) {
                        kt = 0; jt += s_TN;
                        if (jt >= s_Nc) {
                            jt = 0; it += s_TM;
                            if (it >= s_Mc) {
                                layer_done = true;
                                configured = false;
                                global_c_offset += s_Mc * s_Nc;
                            }
                        }
                    }
                }
            }
        }
    }
}