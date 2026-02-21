`timescale 1ns/1ps

module PE_tb;

// Parameters
parameter DATA_W     = 8;
parameter DATA_W_OUT = 32;

// DUT signals
logic                   clk;
logic                   rst_n;
logic [DATA_W-1:0]      in_act;
logic [DATA_W_OUT-1:0]  in_psum;
logic [DATA_W-1:0]      w_in_down;
logic [DATA_W-1:0]      w_in_left;
logic                   load_w;
logic                   transpose_en;

logic [DATA_W-1:0]      out_act;
logic [DATA_W_OUT-1:0]  out_psum;
logic [DATA_W-1:0]      w_out_up;
logic [DATA_W-1:0]      w_out_right;

// Instantiate DUT
PE #(.DATA_W(DATA_W), .DATA_W_OUT(DATA_W_OUT)) dut (
    .clk          (clk),
    .rst_n        (rst_n),
    .in_act       (in_act),
    .in_psum      (in_psum),
    .w_in_down    (w_in_down),
    .w_in_left    (w_in_left),
    .load_w       (load_w),
    .transpose_en (transpose_en),
    .out_act      (out_act),
    .out_psum     (out_psum),
    .w_out_up     (w_out_up),
    .w_out_right  (w_out_right)
);

// Clock: 10ns period
initial clk = 0;
always #5 clk = ~clk;

// Scoreboard / checker
int pass_count = 0;
int fail_count = 0;

task automatic check(
    input string       test_name,
    input logic [DATA_W-1:0]     exp_out_act,
    input logic [DATA_W_OUT-1:0] exp_out_psum,
    input logic [DATA_W-1:0]     exp_w_out_up,
    input logic [DATA_W-1:0]     exp_w_out_right
);
    if (out_act      !== exp_out_act    ||
        out_psum     !== exp_out_psum   ||
        w_out_up     !== exp_w_out_up   ||
        w_out_right  !== exp_w_out_right) begin
        $display("FAIL [%s] @%0t", test_name, $time);
        $display("  out_act:     got %0d, exp %0d", out_act,     exp_out_act);
        $display("  out_psum:    got %0d, exp %0d", out_psum,    exp_out_psum);
        $display("  w_out_up:    got %0d, exp %0d", w_out_up,    exp_w_out_up);
        $display("  w_out_right: got %0d, exp %0d", w_out_right, exp_w_out_right);
        fail_count++;
    end else begin
        $display("PASS [%s]", test_name);
        pass_count++;
    end
endtask

// Helper: apply one rising-edge cycle
task automatic tick;
    @(posedge clk);
    #1; // small settle after clock edge
endtask

// -----------------------------------------------------------------------
// TEST SEQUENCE
// -----------------------------------------------------------------------
initial begin
    $dumpfile("PE_tb.vcd");
    $dumpvars(0, PE_tb);

    // ----- Initialise inputs -----
    rst_n        = 0;
    in_act       = 0;
    in_psum      = 0;
    w_in_down    = 0;
    w_in_left    = 0;
    load_w       = 0;
    transpose_en = 0;

    // ----- TEST 1: Reset check -----
    tick;
    rst_n = 1;
    tick;
    check("Reset", 8'h00, 32'h00000000, 8'h00, 8'h00);

    // ---------------------------------------------------------------
    // TEST 2: Load weight from w_in_down (transpose_en = 0)
    // ---------------------------------------------------------------
    w_in_down    = 8'd5;
    w_in_left    = 8'd99; // should be ignored
    load_w       = 1;
    transpose_en = 0;
    tick;
    // W_reg = 5; w_out_up = 5, w_out_right = 0
    check("Load_w_down", 8'h00, 32'h00000000, 8'd5, 8'h00);

    // ---------------------------------------------------------------
    // TEST 3: Load weight from w_in_left (transpose_en = 1)
    // ---------------------------------------------------------------
    w_in_left    = 8'd7;
    w_in_down    = 8'd99; // should be ignored
    transpose_en = 1;
    tick;
    // W_reg = 7; w_out_up = 0, w_out_right = 7
    check("Load_w_left", 8'h00, 32'h00000000, 8'h00, 8'd7);

    // ---------------------------------------------------------------
    // TEST 4: MAC operation (non-transpose)
    //   W = 5 (reload), in_act = 3, in_psum = 10  => psum = 3*5+10 = 25
    // ---------------------------------------------------------------
    // First reload W=5 in normal mode
    transpose_en = 0;
    w_in_down    = 8'd5;
    load_w       = 1;
    tick;
    check("Reload_W5", 8'h00, 32'h00000000, 8'd5, 8'h00);

    // Now run MAC
    load_w    = 0;
    in_act    = 8'd3;
    in_psum   = 32'd10;
    tick;
    // act_reg = 3, psum_reg = 3*5 + 10 = 25
    check("MAC_basic", 8'd3, 32'd25, 8'd5, 8'h00);

    // ---------------------------------------------------------------
    // TEST 5: Chained MAC
    //   Previous out_psum = 25; feed it back, in_act = 2 => 2*5+25 = 35
    // ---------------------------------------------------------------
    in_psum = 32'd25;
    in_act  = 8'd2;
    tick;
    check("MAC_chain", 8'd2, 32'd35, 8'd5, 8'h00);

    // ---------------------------------------------------------------
    // TEST 6: Zero activation => psum should equal in_psum
    // ---------------------------------------------------------------
    in_act  = 8'd0;
    in_psum = 32'd100;
    tick;
    check("Zero_act", 8'd0, 32'd100, 8'd5, 8'h00);

    // ---------------------------------------------------------------
    // TEST 7: Zero weight
    // ---------------------------------------------------------------
    load_w    = 1;
    w_in_down = 8'd0;
    tick;
    // W=0
    load_w  = 0;
    in_act  = 8'd15;
    in_psum = 32'd42;
    tick;
    // psum = 15*0 + 42 = 42
    check("Zero_weight", 8'd15, 32'd42, 8'd0, 8'h00);

    // ---------------------------------------------------------------
    // TEST 8: Max weight and activation (overflow check, 8-bit unsigned)
    //   W=255, act=255 => 255*255 = 65025; +0 = 65025
    // ---------------------------------------------------------------
    load_w    = 1;
    w_in_down = 8'd255;
    tick;
    load_w  = 0;
    in_act  = 8'd255;
    in_psum = 32'd0;
    tick;
    check("Max_values", 8'd255, 32'd65025, 8'd255, 8'h00);

    // ---------------------------------------------------------------
    // TEST 9: Transpose mode MAC
    //   Load W=4 via w_in_left, then MAC: act=6, psum=1 => 24+1=25
    // ---------------------------------------------------------------
    load_w       = 1;
    transpose_en = 1;
    w_in_left    = 8'd4;
    tick;
    // W=4, w_out_right=4, w_out_up=0
    check("Transpose_load", 8'd255, 32'd65025, 8'h00, 8'd4);

    load_w  = 0;
    in_act  = 8'd6;
    in_psum = 32'd1;
    tick;
    // psum = 6*4 + 1 = 25
    check("Transpose_MAC", 8'd6, 32'd25, 8'h00, 8'd4);

    // ---------------------------------------------------------------
    // TEST 10: load_w blocks MAC update
    //   While load_w=1, act_reg and psum_reg must NOT update
    // ---------------------------------------------------------------
    // Currently: act_reg=6, psum_reg=25, W=4
    load_w  = 1;
    w_in_left = 8'd4; // keep W same, transpose still on
    in_act  = 8'd9;   // should not propagate
    in_psum = 32'd999;
    tick;
    check("Load_w_blocks_MAC", 8'd6, 32'd25, 8'h00, 8'd4);

    // ---------------------------------------------------------------
    // TEST 11: Async reset mid-operation
    // ---------------------------------------------------------------
    load_w  = 0;
    in_act  = 8'd3;
    in_psum = 32'd5;
    @(posedge clk); #1;
    rst_n = 0;       // assert reset asynchronously
    #2;
    // Immediately after reset, registers should be 0
    // (combinational outputs driven from reg, so check now)
    if (out_act !== 0 || out_psum !== 0) begin
        $display("FAIL [Async_reset]: act=%0d psum=%0d", out_act, out_psum);
        fail_count++;
    end else begin
        $display("PASS [Async_reset]");
        pass_count++;
    end
    @(posedge clk); #1;
    rst_n = 1;

    // ---------------------------------------------------------------
    // RANDOMIZED TESTS  TC12 – TC1011  (1000 cases)
    // ---------------------------------------------------------------
    $display("\n--- Randomized Tests (1000 cases) ---");
    begin
        // Working registers — track PE internal state in the TB
        logic [DATA_W-1:0]     W_model;       // mirrors W_reg in DUT
        logic [DATA_W-1:0]     act_model;     // mirrors act_reg
        logic [DATA_W_OUT-1:0] psum_model;    // mirrors psum_reg
        logic                  trans_model;   // mirrors current transpose_en

        logic [DATA_W-1:0]     r_act, r_w_down, r_w_left;
        logic [DATA_W_OUT-1:0] r_psum;
        logic                  r_load_w, r_transpose;

        logic [DATA_W_OUT-1:0] exp_psum;
        logic [DATA_W-1:0]     exp_act;
        logic [DATA_W-1:0]     exp_w_out_up, exp_w_out_right;

        string tname;
        int    rand_pass, rand_fail;
        rand_pass = 0; rand_fail = 0;

        // Bring DUT to clean known state
        rst_n = 1; load_w = 0; transpose_en = 0;
        in_act = 0; in_psum = 0;
        w_in_down = 0; w_in_left = 0;
        tick;

        // Model reset state
        W_model    = '0;
        act_model  = '0;
        psum_model = '0;
        trans_model = '0;

        for (int tc = 12; tc <= 1011; tc++) begin

            // ---- Randomise inputs ----
            r_act       = $urandom();
            r_psum      = $urandom();
            r_w_down    = $urandom();
            r_w_left    = $urandom();
            r_load_w    = $urandom_range(0,1);
            r_transpose = $urandom_range(0,1);

            // Occasionally force interesting corner cases
            case ($urandom_range(0,7))
                0: r_act   = 8'd0;
                1: r_act   = 8'd255;
                2: r_psum  = 32'd0;
                3: r_psum  = 32'hFFFF_FFFF;
                4: begin r_w_down = 8'd0;   r_w_left = 8'd0;   end
                5: begin r_w_down = 8'd255; r_w_left = 8'd255; end
                6: r_load_w = 1;
                7: r_load_w = 0;
            endcase

            // ---- Drive DUT ----
            in_act       = r_act;
            in_psum      = r_psum;
            w_in_down    = r_w_down;
            w_in_left    = r_w_left;
            load_w       = r_load_w;
            transpose_en = r_transpose;
            tick;

            // ---- Update reference model (matches DUT always_ff) ----
            if (r_load_w && !r_transpose)
                W_model = r_w_down;
            else if (r_load_w && r_transpose)
                W_model = r_w_left;

            if (!r_load_w) begin
                act_model  = r_act;
                psum_model = (r_act * W_model) + r_psum;
            end

            trans_model = r_transpose;

            // ---- Compute expected outputs ----
            exp_act         = act_model;
            exp_psum        = psum_model;
            exp_w_out_up    = (trans_model == 0) ? W_model : 8'h00;
            exp_w_out_right = (trans_model)      ? W_model : 8'h00;

            // ---- Check ----
            if (out_act      !== exp_act        ||
                out_psum     !== exp_psum        ||
                w_out_up     !== exp_w_out_up    ||
                w_out_right  !== exp_w_out_right) begin

                $sformat(tname, "RAND_TC%0d", tc);
                $display("FAIL [%s] @%0t", tname, $time);
                $display("  Inputs: load_w=%0b transpose=%0b act=%0d w_down=%0d w_left=%0d psum=%0d",
                         r_load_w, r_transpose, r_act, r_w_down, r_w_left, r_psum);
                $display("  out_act:     got=%0d  exp=%0d", out_act,     exp_act);
                $display("  out_psum:    got=%0d  exp=%0d", out_psum,    exp_psum);
                $display("  w_out_up:    got=%0d  exp=%0d", w_out_up,    exp_w_out_up);
                $display("  w_out_right: got=%0d  exp=%0d", w_out_right, exp_w_out_right);
                fail_count++; rand_fail++;
            end else begin
                pass_count++; rand_pass++;
            end
        end

        $display("  Randomized: %0d PASS  %0d FAIL (out of 1000)", rand_pass, rand_fail);
    end

    // ---------------------------------------------------------------
    // Summary
    // ---------------------------------------------------------------
    $display("\n=============================");
    $display(" Results: %0d PASS  %0d FAIL", pass_count, fail_count);
    $display("=============================\n");

    if (fail_count == 0)
        $display("ALL TESTS PASSED");
    else
        $display("SOME TESTS FAILED");

    $finish;
end

// Timeout watchdog
initial begin
    #200000;
    $display("TIMEOUT");
    $finish;
end

endmodule