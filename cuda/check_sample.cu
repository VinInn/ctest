/*
 * compile with    nvcc -DFOO=acosh -DUSE_FLOAT check_sample.cu --cudart shared -gencode arch=compute_70,code=sm_70 -O3 -std=c++17 --compiler-options="-O3 -lmpfr -lgmp -lm -fopenmp"
 * or _DOUBLE
 */
#include "check_sample.cc"

