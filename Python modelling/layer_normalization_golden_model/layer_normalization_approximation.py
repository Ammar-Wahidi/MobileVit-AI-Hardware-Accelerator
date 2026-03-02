import numpy as np

def layer_norm_approximation(activations_q24_8):
    """
    Fixed-point Layer Normalization hardware approximation model.
    Input: List/Array of 32 elements in Q24.8 (32-bit signed).
    """
    DATA_WIDTH = 32
    
    # 1. Summation and Mean
    # sum(Q24.8) -> Mean (shifted right by 5 for division by 32)
    total_sum = int(np.sum(activations_q24_8))
    mean = total_sum >> 5
    
    # 2. sub_out = activation - mean
    sub_out = [int(x - mean) for x in activations_q24_8]
    
    # 3. Square and Variance
    # (Q24.8 * Q24.8) = Q48.16
    mul_out = [x * x for x in sub_out]
    sum_2 = int(np.sum(mul_out))
    variance = sum_2 >> 5  # sum_2 / 32
    
    # 4. Standard Deviation Inverse logic
    if variance <= 0:
        return [0] * 32
    
    # variance = (2^k) * m
    # k is 6 bits, m is 17 bits (Q1.16)
    v_int = int(variance)
    k = v_int.bit_length() - 17
    
    # Normalize m to 17 bits
    if k >= 0:
        m = v_int >> k
    else:
        m = v_int << abs(k)
        
    # LUT for m[15:8] (Top 8 bits of fractional part)
    # Simulation of the 1/sqrt(m) LUT
    m_float = m / (2**16)
    y0_1 = int((1.0 / np.sqrt(m_float)) * (2**14)) # Q2.14
    
    # 5. Handle Shifter and Constant
    # shifter = k >> 1 if k even, else (k-1) >> 1
    shifter = k >> 1 if (k % 2 == 0) else (k - 1) >> 1
    constant = 0x2D41  # Q2.14 (approx 1/sqrt(2))
    
    # Check k[0] (LSB of k) to see if it's odd
    k_is_odd = bool(k & 1)
    
    if k_is_odd:
        # std_dev_inv is Q4.28
        std_dev_inv = (y0_1 >> shifter) * constant
    else:
        # std_dev_inv is Q18.14
        std_dev_inv = (y0_1 >> shifter)
    
    # 6. Final Normalization
    normalized_output = []
    for i in range(len(sub_out)):
        # multiplication: sub_in[j] (Q24.8) * std_dev_inv
        norm_comb = int(std_dev_inv * sub_out[i])
        
        if k_is_odd:
            # Shift right by 28 bits to get Q24.8
            # Hardware: norm_comb[(28+32-1):28]
            val = (norm_comb >> 28)
        else:
            # Shift right by 14 bits to get Q24.8
            # Hardware: norm_comb[(14+32-1):14]
            val = (norm_comb >> 14)
            
        # Apply 32-bit mask and handle sign extension for Python
        val = val & 0xFFFFFFFF
        if val >= 0x80000000:
            val -= 0x100000000
            
        normalized_output.append(val)
        
    return normalized_output

# --- Simple Test Case ---
# Creating dummy Q24.8 data
test_input = (np.array([10, 20, 30, -10, -20] * 6 + [5, 5]) << 8).astype(np.int64)
result = layer_norm_approximation(test_input)

print(f"Sample Input (Q24.8): {test_input[:5]}")
print(f"Sample Output (Q24.8): {result[:5]}")