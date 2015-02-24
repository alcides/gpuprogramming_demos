nvcc 0_simple.cu  -o 0_simple
nvcc 1_simple_sum.cu  -o 1_simple_sum
nvcc 2_par_sum.cu  -o 2_par_sum
nvcc 3_work_sum.cu -o 3_work_sum
nvcc 4_full_sum.cu -o 4_full_sum
nvcc 5_reduction.cu -o 5_reduction
gcc 6_matrixmult.c -o 6_matrixmult -lOpenCL