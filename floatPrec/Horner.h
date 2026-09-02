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
#define HD_INLINE __device__ __host__ inline
#else
#define HD_INLINE inline
#endif



template<typename Float, int N>
struct Horner {
HD_INLINE Float operator()(Float x, Float const * const c) {
  Horner<Float,N-1> horner;
  #if  defined(__FMA__) || defined(FP_FAST_FMA)
    return std::fma(x,horner(x,c),c[N]);
  #else
    return x*horner(x,c)+c[N];
  #endif
}
};

template<typename Float>
struct Horner<Float,0> {
  HD_INLINE Float operator()(Float x, Float const * const c) { return c[0]; }
};

template<int N, typename Float>
HD_INLINE Float horner(Float x, Float const * const c) { Horner<Float,N> f; return f(x,c);} 
