#ifndef cudaCompat_H
#define cudaCompat_H

#include<cstdint>
#include<algorithm>
#include<cstring>

namespace cudaCompat {

  struct dim3{uint32_t x,y,z;};
  constexpr dim3 threadIdx = {0,0,0};
  constexpr dim3 blockIdx = {0,0,0};
  constexpr dim3 blockDim = {1,1,1};
  constexpr dim3 gridDim = {1,1,1};


  void cudaMemset(void *b, int c, size_t len) {
    memset(b, c, len);
  }

  template<typename T1, typename T2>
  T1  atomicAdd(T1* a, T2 b) {auto ret=*a; (*a) +=b; return ret;}

  template<typename T1, typename T2>
  T1  atomicSub(T1* a, T2 b) {auto ret=*a; (*a) -=b; return ret;}


  template<typename T1, typename T2>
  T1  atomicMin(T1* a, T2 b) {auto ret=*a; *a = std::min(*a,b);return ret;}
  template<typename T1, typename T2>
  T1  atomicMax(T1* a, T2 b) {auto ret=*a; a = std::max(*a,b);return ret;}

  
  void cudaDeviceSynchronize(){}

  void __syncthreads(){}
  bool __syncthreads_or(bool x) { return x;}
  bool __syncthreads_and(bool x) { return x;}
  
}

#ifndef __CUDACC__
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
