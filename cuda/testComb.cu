//nvcc -gencode arch=compute_120,code=sm_120 -O3 --expt-relaxed-constexpr -std=c++23 testComb.cu
///opt/nvidia/nsight-compute/2026.3.1/ncu --metrics sm__cycles_elapsed ./a.out
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
void verify(T const * __restrict__ xi, uint16_t const * __restrict__ ind, int n) {
   int tid = blockDim.x * blockIdx.x + threadIdx.x;

   for (auto i=tid; i<n; i+=gridDim.x * blockDim.x)  {
      if (ind[i] != 15) printf("??? %d %d\n",i,ind[i]);
//      assert(ind[i] == 15);
   }

//  if (threadIdx.x==0) printf("verify ok/n");
}

template<typename T>
__global__
void bfN2(T const * __restrict__ xi, uint16_t * __restrict__ ind,  int n) {
   int tid = blockDim.x * blockIdx.x + threadIdx.x;
   
   for (auto i=tid; i<n; i+=gridDim.x * blockDim.x)  {
    auto x = xi[i];
    auto k = ind[i];
    for (auto j=0; j<n; j++) {
      if (xi[j]>x) { k = j; x=xi[j];}
    }
    ind[i] = k;
   }
}

__global__
void bfN2h2(half2 const * __restrict__ xi, uint32_t * __restrict__ ind,  int n) {
   int tid = blockDim.x * blockIdx.x + threadIdx.x;
   /*
   if (tid==0) {
    half const * h = (half const *)xi;
    uint16_t * j = (uint16_t *)ind;
    printf("--- %.5f %.5f %.5f %.5f %d %d %d %d\n",float(h[14]),float(h[15]),__high2float(xi[7]),__low2float(xi[7]),
      j[14],j[15],ind[7]&0xffff,ind[7]>>16);
   }
   */
   n = n/2;
   for (auto i=tid; i<n; i+=gridDim.x * blockDim.x)  {
    auto x = xi[i];
    auto k = ind[i];
    for (uint32_t j=0; j<n; j++) {
      auto y = xi[j];
      uint32_t j1 = ((j*2+1)<<16) | ((j*2));
      auto m1 = __hgt2_mask(y,x);
      x = __hmax2(x,y);
      k = m1&j1 | (~m1)&k;
      // x = &(half2*)(&(m1&std::bit_cast<uint32_t>(y) | (~m1)&std::bit_cast<uint32_t>(x)));
      uint32_t j2 = ((j*2)<<16) | (j*2+1);
      y = __lowhigh2highlow(y);
      auto m2 = __hgt2_mask(y,x);
      k = m2&j2 | (~m2)&k;
      x = __hmax2(x,y);
      // x = &(half2*)(&(m2&std::bit_cast<uint32_t>(y) | (~m2)&std::bit_cast<uint32_t>(x)));
    }
    ind[i] = k;
   }
}


template<typename T>
__global__
void oneShift(T * __restrict__ xi, uint16_t * __restrict__ ind, int n) {
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
void doShift(T const * __restrict__ xi, uint16_t * __restrict__ ind, int n) {
   int tid = blockDim.x * blockIdx.x + threadIdx.x;
   // int laneId = threadIdx.x & 0x1f;
   int nw = (n+31)/32;
   int ntot = nw*32;
   for (auto i=tid; i<n; i+=gridDim.x * blockDim.x)  {
     auto y = xi[i];
     auto m = ind[i];
     for (int w=0; w<nw; ++w) { 
       int j = tid+32*w;
       if (j>=ntot) j-=ntot;
       T x = xi[j];
       auto k = ind[j];
       for (int t=0; t<32; ++t) {
          auto x0 = __shfl_sync(0xffffffff,x,t);
          auto i0 = __shfl_sync(0xffffffff,k,t);
          if (x0>y) { m = i0; y=x0;}
       } // lanes
     } // warps
     ind[i] = m;
   } // "blocks"
}


#include<random>
template<typename T>
struct G {
  void operator()(int n) {
    cudaMalloc(&a, n * sizeof(T));
    cudaMalloc(&b, n * sizeof(T));
    cudaMalloc(&ind, n * sizeof(uint16_t));

    uint16_t * l = new uint16_t[n];
    T * q = new T[n];

   for (int i=0; i<n;++i) {
     l[i] = i;
     q[i] = i;
   } 
   q[15] = n+3;

    cudaMemcpy(ind, l, n*sizeof(uint16_t),cudaMemcpyHostToDevice);
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
  uint16_t * ind;

};

template<typename T, int NB, int NT>
void doTest(int n) {

  {
    std::cout << "brute force n^2 " << n << std::endl;
    G<T> g;
    g(n);
    if constexpr(std::is_same<half, typename std::remove_cv<T>::type>::value) {
      bfN2h2<<<NB,NT,0,0>>>((half2 const*)g.a,(uint32_t*)g.ind,n);
    } else {
      bfN2<T><<<NB,NT,0,0>>>(g.a,g.ind,n);
    }
#ifdef VERIFY
    verify<T><<<4,128,0,0>>>(g.a,g.ind,n);
#endif
    cudaDeviceSynchronize();
  }
  {
    std::cout << "shift n^2 " << n << std::endl;
    G<T> g;
    g(n);
    doShift<T><<<NB,NT,0,0>>>(g.a,g.ind,n);
#ifdef VERIFY
    verify<T><<<4,128,0,0>>>(g.a,g.ind,n);
#endif
    cudaDeviceSynchronize();
  }
}

int main() {

/*
  doTest<uint16_t,1,64>(64);
  doTest<half,1,64>(64);
  doTest<float,1,64>(64);
*/
  doTest<uint16_t,1,128>(128);
  doTest<half,1,128>(128);
  doTest<float,1,128>(128);


//  doTest<uint16_t,1,128>(1024);
//  doTest<half,1,128>(1024);
 // doTest<float,1,128>(1024);

  return 0;
}
