import os

def read_hex_mem(filename, lines=16):
    with open(filename, 'r') as f:
        return [int(line.strip(), 16) for line in f if line.strip()][:lines]

def read_int_txt_all(filename):
    with open(filename, 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]

def main():
    print("======== Python Batch Norm Mass Verification ========\n")
    
    files_needed = ['A.mem', 'B.mem', 'sa_out.txt', 'rtl_bn_out.txt']
    for file in files_needed:
        if not os.path.exists(file):
            print(f"Error: {file} not found. Run the Verilog simulation first!")
            return

    # Load BN parameters (we only use the first 16 since the wrapper only handles 1 tile)
    A_scale = read_hex_mem('A.mem')
    B_bias  = read_hex_mem('B.mem')
    
    # Load ALL test data
    sa_psum_all = read_int_txt_all('sa_out.txt')
    rtl_out_all = read_int_txt_all('rtl_bn_out.txt')

    N_TILE = 16
    FRAC_BITS = 8
    
    # Calculate how many batches were run
    num_batches = len(sa_psum_all) // N_TILE
    print(f"Detected {num_batches} batches in the output files.\n")

    all_pass = True
    total_errors = 0

    # Print Table Header
    print(f"{'Batch':<6} | {'Idx':<4} | {'SA (Psum)':<12} | {'Py BN Out':<12} | {'RTL BN Out':<12} | {'Status'}")
    print("-" * 75)

    # Process batch by batch
    for batch_idx in range(num_batches):
        start_idx = batch_idx * N_TILE
        end_idx = start_idx + N_TILE
        
        batch_sa = sa_psum_all[start_idx:end_idx]
        batch_rtl = rtl_out_all[start_idx:end_idx]

        for i in range(N_TILE):
            P = batch_sa[i]
            scale = A_scale[i]
            bias = B_bias[i]

            # Python emulation of the hardware math
            term1 = scale * P 
            term2 = bias << FRAC_BITS
            py_bn_out = (term1 + term2) >> FRAC_BITS
            
            # Determine status
            if py_bn_out == batch_rtl[i]:
                status = "PASS"
            else:
                status = "FAIL"
                all_pass = False
                total_errors += 1
                
            # Print EVERY comparison
            print(f"{batch_idx:<6} | {i:<4} | {P:<12} | {py_bn_out:<12} | {batch_rtl[i]:<12} | {status}")

    print("-" * 75)
    if all_pass:
        print(f"SUCCESS: Python matched RTL hardware across all {num_batches * N_TILE} operations! 🎉")
    else:
        print(f"ERROR: Found {total_errors} mismatches.")

if __name__ == "__main__":
    main()