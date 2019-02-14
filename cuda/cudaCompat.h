#ifndef HeterogeneousCore_CUDAUtilities_cudaCompat_H
#define HeterogeneousCore_CUDAUtilities_cudaCompat_H

/*
 * Everything you need to run cuda code in plain sequential c++ code
 */




#ifndef __CUDA_ARCH__

#include<cstdint>
#include<algorithm>
#include<cstring>

#include "cuda_runtime.h"

namespace cudaCompat {

#ifndef __CUDA_RUNTIME_H__
  struct dim3{uint32_t x,y,z;};
#endif
  const dim3 threadIdx = {0,0,0};
  const dim3 blockDim = {1,1,1};
  thread_local dim3 blockIdx = {0,0,0};
  thread_local dim3 gridDim = {1,1,1};



 template<typename T1, typename T2>
 T1  atomicInc(T1* a, T2 b) {auto ret=*a; if ((*a)<b) (*a)++; return ret;}

  
  template<typename T1, typename T2>
  T1  atomicAdd(T1* a, T2 b) {auto ret=*a; (*a) +=b; return ret;}

  template<typename T1, typename T2>
  T1  atomicSub(T1* a, T2 b) {auto ret=*a; (*a) -=b; return ret;}


  template<typename T1, typename T2>
  T1  atomicMin(T1* a, T2 b) {auto ret=*a; *a = std::min(*a,b);return ret;}
  template<typename T1, typename T2>
  T1  atomicMax(T1* a, T2 b) {auto ret=*a; a = std::max(*a,b);return ret;}

  
  void __syncthreads(){}
  bool __syncthreads_or(bool x) { return x;}
  bool __syncthreads_and(bool x) { return x;}

  void resetGrid() {
    blockIdx = {0,0,0};
    gridDim = {1,1,1};
  }
  
}

#ifndef __CUDA_RUNTIME_H__
#define __host__
#define __device__
#define __global__
#define __shared__
#define __forceinline__
#endif


#ifndef __CUDA_ARCH__
using namespace cudaCompat;
#endif

#endif


#endif
