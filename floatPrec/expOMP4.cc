#include "approx_exp.h"



#define DEGREE 5

void bar(float * b, float const * a, int NN) {
#pragma omp simd aligned(a, b: 32)
   for (int i=0; i<NN; ++i)
      b[i] = approx_expf<DEGREE>(a[i]);
}


