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

#include<cstdio>

 __device__ uint32_t hashedIndexEE(uint32_t id);

__constant__ int q[] = {0,1,2,3,4};

__device__  __forceinline__
void barf(int * i) {
  extern __shared__  unsigned char shared_mem[];
  shared_mem[threadIdx.x]=i[threadIdx.x]*q[2]*hashedIndexEE(i[7]);
  __syncthreads();
  printf("bar %d %d\n", shared_mem[0],q[3]);
  int * q = 0;
  if (i[2]>3) {
    q = new int[i[2]];
    memset(q,0,4*i[2]);
    atomicAdd(q+2,shared_mem[0]);
    memcpy(i,q,min(40,4*i[2]));
    __syncthreads();
    i[4] = q[threadIdx.x];
  }
  delete [] q;
}



__global__
void bar(int * i) {
  barf(i);
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



__global__
void crash(int * i) {
  bar<<<1,1>>>(i);
  cudaDeviceSynchronize();
}



/*
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__
void coop(int * i) {

  grid_group grid = this_grid();

  barf(i);
  grid.sync();
  barf(i);
  grid.sync();

}
*/


#include "cudaCheck.h"
void wrapper() {
  int a[10]; a[0]=4; a[2]=0;
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
