#include <hip/hip_runtime.h>
#include<cmath>
#include<cstdio>


__global__ void doit(float x) {
   printf ("%f %f\n",x, cosf(x));
  // *y = sqrtf(x);

}


int main() {

  hipStream_t stream;
  hipStreamCreate(&stream);

  doit<<<1,1,0,stream>>>(0.34);

  hipStreamSynchronize(stream);

  return 0;

}
