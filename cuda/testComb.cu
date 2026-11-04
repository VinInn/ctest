#include <cassert>
#include <cstdint>
#include <cmath>
#include <cuda_fp16.h>
// #include <crt/device_fp128_functions.h>
#include <bit>
#include <cstdio>
#include <iostream>

#ifndef HOST_DEVICE_CONSTANT
#ifdef __CUDA_ARCH__
#define HOST_DEVICE_CONSTANT __device__ constexpr
#else
#define HOST_DEVICE_CONSTANT constexpr
#endif
#endif

#ifdef __NVCC__
#define HD_INLINE __device__ __host__ inline
#define D_INLINE __device__ __forceinline__
#else
#define HD_INLINE inline
#define D_INLINE inline
#endif

template<typename T>
__global__
void verify(T const * __restrict__ xi, int const * __restrict__ ind, int n) {
   int tid = blockDim.x * blockIdx.x + threadIdx.x;

   for (auto i=tid; i<n; i+=gridDim.x * blockDim.x)  {
      assert(ind[i] == n-1);
   }

//  if (threadIdx.x==0) printf("verify ok/n");
}

template<typename T>
__global__
void btN2(T const * __restrict__ xi, int * __restrict__ ind,  int n) {
   int tid = blockDim.x * blockIdx.x + threadIdx.x;
   
   for (auto i=tid; i<n; i+=gridDim.x * blockDim.x)  {
    for (auto j=i; j<n; j++) {
      if (xi[j]>xi[i]) ind[i] = j;
    }
   }

}



template<typename T>
__global__
void oneShift(T * __restrict__ xi, int * __restrict__ ind, int n) {
   int tid = blockDim.x * blockIdx.x + threadIdx.x;
   T x = xi[tid];
   int k = ind[tid];
   int m = __shfl_sync(0xffffffff,k,0,32);
   __syncthreads();
   printf("su1  %d %d %d %d\n",tid, ind[tid],k,m);
   m = __shfl_sync(0xffffffff,k,15,32);
   __syncthreads();
   printf("su2  %d %d %d %d\n",tid, ind[tid],k,m);
   m = __shfl_sync(0xffffffff,k,3,32);
   __syncthreads();
   printf("su3  %d %d %d %d\n",tid, ind[tid],k,m);
}

template<typename T>
__global__ 
void doShift(T * __restrict__ xi, int * __restrict__ ind, int n) {
   int tid = blockDim.x * blockIdx.x + threadIdx.x;
   // int laneId = threadIdx.x & 0x1f;
   int nw = (n+31)/32;
   int ntot = nw*32;
   for (auto i=tid; i<n; i+=gridDim.x * blockDim.x)  {
     for (int w=0; w<nw; ++w) { 
       int j = tid+32*w;
       if (j>=ntot) j-=ntot;
       T x = xi[j];
       int k = ind[j];
       for (int t=0; t<32; ++t) {
          auto x0 = __shfl_sync(0xffffffff,x,t);
          auto i0 = __shfl_sync(0xffffffff,k,t);
          if (x0>xi[i]) ind[i] = i0;
       } // lanes
     } // warps
   } // "blocks"
}


#include<random>
template<typename T>
struct G {
  void operator()(int n) {
    cudaMalloc(&a, n * sizeof(T));
    cudaMalloc(&b, n * sizeof(T));
    cudaMalloc(&ind, n * sizeof(int));

    int * l = new int[n];
    T * q = new T[n];

   for (int i=0; i<n;++i) {
     l[i] = i;
     q[i] = i;
   } 

    cudaMemcpy(ind, l, n*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(a, q, n*sizeof(T),cudaMemcpyHostToDevice);
    delete [] l;
    delete [] q;

  }
  ~G() {
    cudaFree(a);
    cudaFree(b);
    cudaFree(ind);
  }

  T * a;
  T * b;
  int * ind;

};

template<typename T, int NB, int NT>
void doTest(int n) {

  {
    std::cout << "brute force n^2 " << n << std::endl;
    G<T> g;
    g(n);
    btN2<T><<<NB,NT,0,0>>>(g.a,g.ind,n);
    verify<T><<<128,4,0,0>>>(g.a,g.ind,n);
    cudaDeviceSynchronize();
  }
  {
    std::cout << "shift n^2 " << n << std::endl;
    G<T> g;
    g(n);
    doShift<T><<<NB,NT,0,0>>>(g.a,g.ind,n);
    verify<T><<<128,4,0,0>>>(g.a,g.ind,n);
    cudaDeviceSynchronize();
  }
}

int main() {


  doTest<float,1,64>(64);
  // doTest<float,1,64>(1024);


  return 0;
}
