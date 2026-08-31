#pragma once
#include<cmath>

#ifndef HOST_DEVICE_CONSTANT
#ifdef __CUDA_ARCH__
#define HOST_DEVICE_CONSTANT __device__ constexpr
#else
#define HOST_DEVICE_CONSTANT constexpr
#endif
#endif

#ifdef __NVCC__
#define HD_INLINE __device__ __host__ constexpr
#else
#define HD_INLINE inline constexpr
#endif



template<int N>
HD_INLINE float horner(float x, float const * const c) {
#if  defined(__FMA__) || defined(FP_FAST_FMA)
  return std::fma(x,horner<N-1>(x,c),c[N]);
#else
  return x*horner<N-1>(x,c)+c[N];
#endif
}

template<>
inline
HD_INLINE float horner<0>(float x, float const * const c) {
  return c[0];
}
