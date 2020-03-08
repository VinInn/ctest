#include <cuda.h>
#include <cuda_runtime.h>
#include<cassert>
struct Go {

  virtual ~Go(){}

  virtual void go() =0;

  inline
  static Go * me(Go * in=nullptr) {
    static Go * l = nullptr;
    if (in) l=in;
    return l;
  }

};


#include "cudaCheck.h"

void alloc(int i) {
  int * a = new int[i]; a[0]=4;
  int * d;
  cudaMalloc(&d,4*i);
  cudaMemcpyAsync(d,a,4*i,cudaMemcpyHostToDevice,0);
  cudaCheck(cudaGetLastError());
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
}


#include<iostream>
struct Me2 : private Go {

  Me2() {
   std::cout << "Loaded Me2" << std::endl;
   alloc(1024);
   // assert(this==me(this));
  }


  void go() override { alloc(1024);    std::cout << "Me2 go" << std::endl; }

};


Me2 me2;
