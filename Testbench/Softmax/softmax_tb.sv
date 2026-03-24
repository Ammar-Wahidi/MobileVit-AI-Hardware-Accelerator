`timescale 1ns/1ps

module softmax_tb;

    parameter DATA_WIDTH  = 32;
    parameter FRAC_BITS   = 16;
    parameter VECTOR_SIZE = 8;

    logic signed [DATA_WIDTH-1:0] input_vector  [VECTOR_SIZE];
    logic signed [DATA_WIDTH-1:0] output_vector [VECTOR_SIZE];

    softmax_top #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .VECTOR_SIZE(VECTOR_SIZE)
    ) dut (
        .input_vector(input_vector),
        .output_vector(output_vector)
    );

    //-----------------------------------------
    // Utilities
    //-----------------------------------------

    function real fixed_to_real(input logic signed [DATA_WIDTH-1:0] val);
        fixed_to_real = val;
        fixed_to_real = fixed_to_real / (1 << FRAC_BITS);
    endfunction

    function logic signed [DATA_WIDTH-1:0] real_to_fixed(input real r);
        real_to_fixed = r * (1 << FRAC_BITS);
    endfunction

    //-----------------------------------------
    // Print vector
    //-----------------------------------------

    task print_vector(input string title,
                      input logic signed [DATA_WIDTH-1:0] vec[VECTOR_SIZE]);

        int i;
        real v;

        $display("\n%s", title);

        for (i = 0; i < VECTOR_SIZE; i++) begin
            v = fixed_to_real(vec[i]);
            $display("  [%0d] = %f", i, v);
        end
    endtask

    //-----------------------------------------
    // Check sum of softmax
    //-----------------------------------------

    task check_softmax_sum();

        real sum;
        int i;

        sum = 0;

        for (i=0;i<VECTOR_SIZE;i++)
            sum += fixed_to_real(output_vector[i]);

        $display("\nSoftmax Sum = %f", sum);

        if (sum > 1.1 || sum < 0.9)
            $display("WARNING: Sum not close to 1");
        else
            $display("PASS: Softmax normalization OK");

    endtask

    //-----------------------------------------
    // Check negative outputs
    //-----------------------------------------

    task check_negative();

        int i;

        for (i=0;i<VECTOR_SIZE;i++) begin
            if (output_vector[i] < 0)
                $display("WARNING: negative softmax at index %0d", i);
        end

    endtask

    //-----------------------------------------
    // Apply vector
    //-----------------------------------------

    task apply_vector(input real values[VECTOR_SIZE]);

        int i;

        for (i=0;i<VECTOR_SIZE;i++)
            input_vector[i] = real_to_fixed(values[i]);

        #100;

        print_vector("Input Vector", input_vector);
        print_vector("Softmax Output", output_vector);

        check_softmax_sum();
        check_negative();

    endtask

    //-----------------------------------------
    // Random vector generator
    //-----------------------------------------

    task random_test();

        real rand_vec[VECTOR_SIZE];
        int i;

        for (i=0;i<VECTOR_SIZE;i++)
           rand_vec[i] = $urandom_range(0,10) - 5;

        apply_vector(rand_vec);

    endtask

    //-----------------------------------------
    // Directed tests
    //-----------------------------------------

    task automatic directed_tests();

        real vec1[VECTOR_SIZE] = '{1,2,3,4,1,2,3,4};
        real vec2[VECTOR_SIZE] = '{0,0,0,0,0,0,0,0};
        real vec3[VECTOR_SIZE] = '{5,1,0,-1,-2,-3,-4,-5};
        real vec4[VECTOR_SIZE] = '{10,9,8,7,6,5,4,3};

        $display("\n===== Directed Test 1 =====");
        apply_vector(vec1);

        $display("\n===== Directed Test 2 =====");
        apply_vector(vec2);

        $display("\n===== Directed Test 3 =====");
        apply_vector(vec3);

        $display("\n===== Directed Test 4 =====");
        apply_vector(vec4);

    endtask

    //-----------------------------------------
    // Random regression
    //-----------------------------------------

    task random_regression();

        int i;

        $display("\n===== Random Regression =====");

        for (i=0;i<20;i++) begin
            $display("\nRandom Test %0d", i);
            random_test();
        end

    endtask

    //-----------------------------------------
    // Main test
    //-----------------------------------------

    initial begin

        $display("\n=====================================");
        $display("        Softmax Verification");
        $display("=====================================\n");

        directed_tests();

        random_regression();

        $display("\nSimulation Finished Successfully");

        #100;
        $finish;

    end

endmodule