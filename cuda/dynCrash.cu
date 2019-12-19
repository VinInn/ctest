#include <cuda.h>
#include <cuda_runtime.h>


#include<cstdio>

__global__
void bar(int i) {
  extern __shared__  unsigned char shared_mem[];
  shared_mem[threadIdx.x]=1;
  __syncthreads();
  printf("bar %d\n", shared_mem[0]);
}


/*
__global__
void crash() {
  bar<<<1,1>>>();
  cudaDeviceSynchronize();
}
*/

#include "cudaCheck.h"
void wrapper() {
  bar<<<1,1,1024,0>>>(1);
  cudaCheck(cudaGetLastError());
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
}


#include<iostream>
struct Me {

  Me() {
   std::cout << "Loaded" << std::endl;
   wrapper();
  }

};


Me me;
