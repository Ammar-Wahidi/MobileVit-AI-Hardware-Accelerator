module swish #(
    parameter int WIDTH = 16,        
    parameter int FRACT_BITS = 8    
)(
    input  wire signed [WIDTH-1:0] x,
    output wire signed [WIDTH-1:0] y
);


    localparam signed [WIDTH-1:0] CONST_3 = 3 << FRACT_BITS;
    localparam signed [WIDTH-1:0] CONST_6 = 6 << FRACT_BITS;

    // 2. x + 3 and ReLU6
    wire signed [WIDTH-1:0] x_plus3;
    wire signed [WIDTH-1:0] relu6_val;

    assign x_plus3   = x + CONST_3;
    assign relu6_val = (x_plus3 < 0) ? 0 : (x_plus3 > CONST_6) ? CONST_6 : x_plus3;

    wire signed [2*WIDTH-1:0] product;
    assign product = x * relu6_val;

    

    wire signed [2*WIDTH-1:0] shift_term_0;
    wire signed [2*WIDTH-1:0] shift_term_1;
    wire signed [2*WIDTH-1:0] shift_term_2;
    wire signed [2*WIDTH-1:0] shift_term_3;
    wire signed [2*WIDTH-1:0] shift_term_4;
    wire signed [2*WIDTH-1:0] shift_term_5;

    assign shift_term_0 = { {3{product[2*WIDTH-1]}},  product[2*WIDTH-1 : 3] };
    assign shift_term_1 = { {5{product[2*WIDTH-1]}},  product[2*WIDTH-1 : 5] };
    assign shift_term_2 = { {7{product[2*WIDTH-1]}},  product[2*WIDTH-1 : 7] };
    assign shift_term_3 = { {9{product[2*WIDTH-1]}},  product[2*WIDTH-1 : 9] };
    assign shift_term_4 = { {11{product[2*WIDTH-1]}}, product[2*WIDTH-1 : 11] };
    assign shift_term_5 = { {13{product[2*WIDTH-1]}}, product[2*WIDTH-1 : 13] };

    // 5. Accumulate 
    wire signed [2*WIDTH-1:0] sum_approx;
    
    assign sum_approx = shift_term_0 + 
                        shift_term_1 + 
                        shift_term_2 + 
                        shift_term_3 + 
                        shift_term_4 + 
                        shift_term_5;

    assign y = sum_approx[WIDTH + FRACT_BITS - 1 : FRACT_BITS];

endmodule