#include "emptyKernel.h"

/*
__global__ void foo(int *x, int  *y) {
  int a[2] = {1,2};
#ifndef __CUDA_ARCH__
#warning Host mode
   auto [*x,*y] = a;
#else
  *x = a[0]; *y=a[1];
#endif
}
*/

void a() {}
