#include <cstdio>
__global__ 
void foo() {
   printf("Hello\n");
}

#include<cuda.h>
#include "cudaCheck.h"

void fooWrapper() {

  foo<<<1,1>>>();
  cudaCheck(cudaGetLastError());
  cudaDeviceSynchronize();
 

}


void docheck() {
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
}
