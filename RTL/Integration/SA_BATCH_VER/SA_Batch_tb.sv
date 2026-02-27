`timescale 1ns/1ps

module SA_Tile_BN_Wrapper_tb;

    // Parameters matching our Q24.8 assumption
    localparam DATA_W     = 8;
    localparam DATA_W_OUT = 32;
    localparam FRAC_BITS  = 8; 
    localparam N_TILE     = 16;
    localparam CLK_PERIOD = 10;
    
    // Number of random batches to run
    localparam NUM_TESTS  = 1000; 

    // DUT Signals (Strictly unsigned)
    logic                      clk;
    logic                      rst_n;
    logic [DATA_W-1:0]         act_in       [N_TILE];
    logic [DATA_W-1:0]         weight_in    [N_TILE];
    logic                      transpose_en;
    logic                      load_w;
    logic                      sa_valid_out;
    
    // BN Parameters
    logic [DATA_W_OUT-1:0]     bn_scale     [0:N_TILE-1];
    logic [DATA_W_OUT-1:0]     bn_bias      [0:N_TILE-1];
    logic [DATA_W_OUT-1:0]     bn_out_row   [0:N_TILE-1];
    logic                      bn_valid_out;

    // Increased array sizes to absorb empty lines in .mem files
    logic [15:0] A_file_data [0:511];
    logic [15:0] B_file_data [0:511];
    
    // Arrays for Randomized Inputs and Expected Results
    logic [DATA_W-1:0]         rand_W        [N_TILE][N_TILE];
    logic [DATA_W-1:0]         rand_A        [N_TILE];
    logic [DATA_W_OUT-1:0]     expected_psum [N_TILE];
    logic [DATA_W_OUT-1:0]     expected_out  [N_TILE];

    // File descriptors and loop variables
    integer fd_sa; 
    integer fd_bn; 
    integer i, j, k, test_idx;
    integer errors = 0;

    // Clock Generation
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // Instantiate the Wrapper
    SA_Tile_BN_Wrapper #(
        .DATA_W(DATA_W),
        .DATA_W_OUT(DATA_W_OUT),
        .FRAC_BITS(FRAC_BITS),
        .N_TILE(N_TILE)
    ) DUT (
        .clk(clk),
        .rst_n(rst_n),
        .act_in(act_in),
        .weight_in(weight_in),
        .transpose_en(transpose_en),
        .load_w(load_w),
        .sa_valid_out(sa_valid_out),
        .bn_scale(bn_scale),
        .bn_bias(bn_bias),
        .bn_out_row(bn_out_row),
        .bn_valid_out(bn_valid_out)
    );

    initial begin
        $display("=========================================================");
        $display("  Starting MASSIVE RANDOMIZED Sweep (%0d Batches)", NUM_TESTS);
        $display("=========================================================\n");
        
        $readmemh("A.mem", A_file_data);
        $readmemh("B.mem", B_file_data);

        // Clear previous output files before the loop starts
        fd_sa = $fopen("sa_out.txt", "w"); $fclose(fd_sa);
        fd_bn = $fopen("rtl_bn_out.txt", "w"); $fclose(fd_bn);

        for (test_idx = 0; test_idx < NUM_TESTS; test_idx++) begin
            
            // 1. Reset System for clean pipeline state
            rst_n = 0;
            transpose_en = 0;
            load_w = 0;
            sa_valid_out = 0;
            #(2*CLK_PERIOD);
            rst_n = 1;
            #(CLK_PERIOD);

            // 2A. Generate ALL random matrices first so they are fully populated
            for (i = 0; i < N_TILE; i++) begin
                rand_A[i] = $urandom_range(1, 100); 
                for (j = 0; j < N_TILE; j++) begin
                    rand_W[i][j] = $urandom_range(1, 100);
                end
            end

            // 2B. Now map params and calculate expected mathematics
            for (i = 0; i < N_TILE; i++) begin
                act_in[i]    = '0;
                weight_in[i] = '0;
                bn_scale[i]  = {16'h0000, A_file_data[i]}; 
                bn_bias[i]   = {16'h0000, B_file_data[i]}; 
                
                expected_psum[i] = 0;
                for (j = 0; j < N_TILE; j++) begin
                    // Now rand_W[j][i] is guaranteed to hold a valid number
                    expected_psum[i] += rand_A[j] * rand_W[j][i];
                end
                
                expected_out[i] = ((bn_scale[i] * expected_psum[i]) + (bn_bias[i] << FRAC_BITS)) >> FRAC_BITS;
            end

            // 3. Load Weights
            load_w = 1;
            for (k = 0; k < N_TILE; k++) begin
                for (i = 0; i < N_TILE; i++) weight_in[i] = rand_W[k][i]; 
                @(posedge clk);
            end
            load_w = 0;
            for (i = 0; i < N_TILE; i++) weight_in[i] = '0;

            // 4. Feed Activations
            for (i = 0; i < N_TILE; i++) act_in[i] = rand_A[i]; 
            @(posedge clk);
            for (i = 0; i < N_TILE; i++) act_in[i] = '0;

            // 5. Trigger valid and wait for data
            sa_valid_out = 1;
            
            for (k = 0; k < 64; k++) begin
                @(posedge clk);
                
                if (DUT.sa_psum[0] !== 32'd0 && DUT.sa_psum[0] !== 32'dx) begin
                    
                    // Dump SA Psum
                    fd_sa = $fopen("sa_out.txt", "a");
                    for (i = 0; i < N_TILE; i++) $fdisplay(fd_sa, "%0d", DUT.sa_psum[i]);
                    $fclose(fd_sa);
                    
                    @(posedge clk); // Wait 1 cycle for BN math
                    
                    // Check and Dump BN Out
                    fd_bn = $fopen("rtl_bn_out.txt", "a");
                    for (i = 0; i < N_TILE; i++) begin
                        if (bn_out_row[i] !== expected_out[i]) begin
                            errors++;
                            // Print details for the first few errors to help debug if needed
                            if (errors <= 5) 
                                $display("    Mismatch in Batch %0d, Idx %0d: Exp=%0d, Got=%0d", test_idx, i, expected_out[i], bn_out_row[i]);
                        end
                        $fdisplay(fd_bn, "%0d", bn_out_row[i]);
                    end
                    $fclose(fd_bn);
                    
                    // Pipeline Drain
                    // De-assert valid and wait 50 cycles for all old data to safely exit the 16x16 array 
                    // before looping around to start the next random batch
                    sa_valid_out = 0;
                    for (int flush = 0; flush < 50; flush++) @(posedge clk);
                    
                    break;
                end
            end
            
            if ((test_idx + 1) % 10 == 0)
                $display("  ... Completed %0d / %0d batches", test_idx + 1, NUM_TESTS);

        end

        // 6. Final Summary
        $display("\n=========================================================");
        if (errors == 0)
            $display("  ALL %0d TESTS PASSED with NO MISMATCHES! 🎉", NUM_TESTS);
        else
            $display("  TEST FAILED: Found %0d mismatches.", errors);
        $display("=========================================================\n");

        $finish;
    end

endmodule