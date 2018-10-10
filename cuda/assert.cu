#include<cassert>
#include "cudaCheck.h"

__global__
void bar(int n) {

  __shared__ int x[1024];
  assert(threadIdx.x<n);
  x[threadIdx.x]=n;
  printf("hello %d\n", x[4]);
}




int main(){

  bar<<<1,1024,0>>>(2000);
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
  return 0;
}
