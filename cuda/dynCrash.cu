#include <cuda.h>
#include <cuda_runtime.h>


#include<cstdio>

__global__
void bar() {
  printf("bar\n");
}

__global__
void crash() {
  bar<<<1,1>>>();
  cudaDeviceSynchronize();
}


#include "cudaCheck.h"
void wrapper() {
//  bar<<<1,1>>>();
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
