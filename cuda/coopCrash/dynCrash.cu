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


__constant__ int q[] = {0,1,2,3,4};


#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__
void coop(int * i, int * j) {

  grid_group grid = this_grid();

  i[threadIdx.x] = q[blockIdx.x];
  grid.sync();
  j[threadIdx.x+blockIdx.x] = i[threadIdx.x+blockIdx.x];

}


__global__
void bar(int * i) {
  printf("bar %d\n", q[3]);
  if (i[2]>3) {
    i[4] = q[threadIdx.x];
  }
}

#include "cudaCheck.h"
void wrapper() {
  int a[10]; a[0]=4; a[2]=0;
  int * d;
  cudaMalloc(&d,40);
  cudaMemcpyAsync(d,a,40,cudaMemcpyHostToDevice,0);
  bar<<<1,1,0,0>>>(d);
  cudaCheck(cudaGetLastError());
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
}


#include<iostream>
struct Me : private Go {

  Me(int a) {
     std::cout << "Loaded " << a << std::endl;
     assert(this==Go::me(this));
     assert(this==Go::me());
  }

  void go() override {
   std::cout << "go" << std::endl;
   wrapper();
   std::cout << "gone" << std::endl;
  }

};


Me me(3);

struct QQ {
  QQ() {
    std::cerr << "QQ Loaded"<< std::endl;
   }

};


QQ qq;
