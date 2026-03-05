import numpy as np

def layer_norm_golden_model(activation_in, embed_dim=32):
    """
    Golden model for layer_normalization_top.v
    Matches the flow: Sum -> Mean -> Sub/Mul -> Sum2 -> Var -> StdDevInv -> Norm
    """
    
    # 1. ELEMENTS_SUM & MEAN
    # Hardware: sum_1_out -> mean
    sum_1 = np.sum(activation_in)
    mean = sum_1 // embed_dim  # Using floor division to mimic integer hardware
    
    # 2. array_multiplier_subtractor
    # Hardware: sub_out = (x - mean), mul_out = (x - mean)^2
    sub_out = np.array([x - mean for x in activation_in])
    mul_out = np.array([s**2 for s in sub_out])
    
    # 3. ELEMENTS_SUM_2 & VARI
    # Hardware: sum_2_out -> vari (variance)
    sum_2 = np.sum(mul_out)
    variance = sum_2 // embed_dim
    
    # 4. standard_deviation_inv
    # Hardware calculates: 1 / sqrt(variance + epsilon)
    # Note: If your hardware uses a specific bit-shift (k), 
    # you may need to adjust this part.
    epsilon = 0 # Common in LayerNorm, check if your RTL uses one
    std_dev = np.sqrt(variance + epsilon)
    std_dev_inv = 1.0 / std_dev
    
    # 5. NORMALIZATION_OUT
    # Hardware: (x - mean) * std_dev_inv
    # In your RTL, 'k' is likely a scaling factor for fixed-point precision
    normalized_output = sub_out * std_dev_inv
    
    return normalized_output, mean, variance

# --- Example Usage ---
if __name__ == "__main__":
    # Mock input: 32 elements (EMBED_DIM)
    np.random.seed(42)
    input_data = np.random.randint(-128, 127, size=32)
    
    output, m, v = layer_norm_golden_model(input_data)
    
    print(f"Input Mean: {m}")
    print(f"Input Variance: {v}")
    print(f"First 5 Normalized Outputs: {output[:5]}")