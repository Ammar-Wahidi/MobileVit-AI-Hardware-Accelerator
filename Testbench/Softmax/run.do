quit -sim
vlib work
vlog -f softmax_list.LIST
vsim -voptargs=+acc work.softmax_tb

add wave -position insertpoint  \
sim:/softmax_tb/DATA_WIDTH \
sim:/softmax_tb/FRAC_BITS \
sim:/softmax_tb/input_vector \
sim:/softmax_tb/output_vector \
sim:/softmax_tb/VECTOR_SIZE

run -all
