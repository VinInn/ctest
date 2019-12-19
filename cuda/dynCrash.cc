#include <cuda.h>
#include <cuda_runtime.h>


#include "cudaCheck.h"

void alloc() {
  int a[10]; a[0]=4;
  int * d;
  cudaMalloc(&d,40);
  cudaMemcpyAsync(d,a,40,cudaMemcpyHostToDevice,0);
  cudaCheck(cudaGetLastError());
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
}


#include<iostream>
struct Me2 {

  Me2() {
   std::cout << "Loaded" << std::endl;
   alloc();
  }

};


Me2 me2;
