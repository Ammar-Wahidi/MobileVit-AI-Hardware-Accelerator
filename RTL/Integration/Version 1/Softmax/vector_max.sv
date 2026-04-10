module vector_max #(
    parameter int DATA_WIDTH = 32,
    parameter int VECTOR_SIZE = 8
)
(
    input  logic signed [DATA_WIDTH-1:0] vec [VECTOR_SIZE],
    output logic signed [DATA_WIDTH-1:0] max_value
);

    integer i;

    always_comb begin
        max_value = vec[0];

        for (i = 1; i < VECTOR_SIZE; i++) begin
            if (vec[i] > max_value)
                max_value = vec[i];
        end
    end

endmodule