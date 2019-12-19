#include <cuda.h>
#include <cuda_runtime.h>


#include<cstdio>

__global__
void bar(int * i) {
  extern __shared__  unsigned char shared_mem[];
  shared_mem[threadIdx.x]=i[threadIdx.x];
  __syncthreads();
  printf("bar %d\n", shared_mem[0]);
}

struct Large {
  int v[100];
};

__global__
void huge(int * i,
  int * a1,
  int * a2,
  int * a3,
  int * a4,
  int * a5,
  int * a6,
  int * a7,
  int * a8,
  Large l1, Large l2, Large l3
) {
  extern __shared__  unsigned char shared_mem[];
  shared_mem[threadIdx.x]=i[threadIdx.x];
  __syncthreads();
  printf("bar %d %d\n", shared_mem[0], l1.v[3]);
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
  int a[10]; a[0]=4;
  int * d;
  cudaMalloc(&d,40);
  cudaMemcpyAsync(d,a,40,cudaMemcpyHostToDevice,0);
  bar<<<1,1,1024,0>>>(d);
  cudaCheck(cudaGetLastError());
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
  Large l1, l2, l3;
  l1.v[3]=5;
  huge<<<1,1,1024>>>(d, d,d,d,d, d,d,d,d,l1,l2,l3);
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
