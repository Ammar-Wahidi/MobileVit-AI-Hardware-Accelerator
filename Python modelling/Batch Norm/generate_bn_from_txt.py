import numpy as np
import re
import os

# =====================================================
# CONFIGURATION
# =====================================================

TXT_FILE = r"F:\Courses\STM\bn_running_pretrained.txt"

# 🔴 CHANGE THIS to the BN layer you want
LAYER_NAME = "mobilevit.conv_stem.normalization"

EPS = 1e-5
FRAC_BITS = 8
SCALE = 1 << FRAC_BITS

# =====================================================
# CHECK FILE
# =====================================================

if not os.path.exists(TXT_FILE):
    print("ERROR: File not found:", TXT_FILE)
    exit()

# =====================================================
# FUNCTION TO EXTRACT ARRAY
# =====================================================

def extract_array(text, key):
    pattern = rf"{re.escape(key)}.*?\[(.*?)\]"
    match = re.search(pattern, text, re.DOTALL)
    
    if not match:
        raise ValueError(f"{key} not found in file")
    
    numbers = match.group(1).split(',')
    return np.array([float(x.strip()) for x in numbers])

# =====================================================
# READ FILE
# =====================================================

with open(TXT_FILE, "r") as f:
    content = f.read()

# =====================================================
# EXTRACT PARAMETERS
# =====================================================

gamma = extract_array(content, f"{LAYER_NAME}.weight")
beta  = extract_array(content, f"{LAYER_NAME}.bias")
mean  = extract_array(content, f"{LAYER_NAME}.running_mean")
var   = extract_array(content, f"{LAYER_NAME}.running_var")

print("Gamma:", gamma)
print("Beta :", beta)
print("Mean :", mean)
print("Var  :", var)

# =====================================================
# COMPUTE A AND B
# =====================================================

A = gamma / np.sqrt(var + EPS)
B = beta - (gamma * mean) / np.sqrt(var + EPS)

print("\nA (float):", A)
print("B (float):", B)

# =====================================================
# CONVERT TO Q8.8
# =====================================================

A_fixed = np.round(A * SCALE).astype(np.int16)
B_fixed = np.round(B * SCALE).astype(np.int16)

print("\nA_fixed:", A_fixed)
print("B_fixed:", B_fixed)

# =====================================================
# SAVE HEX FILES
# =====================================================

np.savetxt("A.mem", A_fixed & 0xFFFF, fmt="%04X")
np.savetxt("B.mem", B_fixed & 0xFFFF, fmt="%04X")

print("\nSUCCESS ✅")
print("Generated A.mem and B.mem")
