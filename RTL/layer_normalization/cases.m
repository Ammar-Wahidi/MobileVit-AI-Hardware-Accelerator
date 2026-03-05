EMBED_DIM = 32;
NUM_TESTS = 1000;
FRAC_BITS = 8;
SCALE = 2^FRAC_BITS;
EPSILON = 1e-5;

activation_file = fopen('activation_in.txt','w');
output_file = fopen('normalized_out.txt','w');

for t = 1:NUM_TESTS

    activation = -5 + 10*rand(1, EMBED_DIM);

    mean_val = mean(activation);
    variance = mean((activation - mean_val).^2);
    std_inv = 1 / sqrt(variance + EPSILON);

    normalized = (activation - mean_val) * std_inv;

    activation_fixed = round(activation * SCALE);
    normalized_fixed = round(normalized * SCALE);

    for i = 1:EMBED_DIM
        fprintf(activation_file,'%08X\n', typecast(int32(activation_fixed(i)), 'uint32'));
        fprintf(output_file,'%08X\n', typecast(int32(normalized_fixed(i)), 'uint32'));
    end
end

fclose(activation_file);
fclose(output_file);

disp('Test vectors generated successfully!');
