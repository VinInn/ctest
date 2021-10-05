#include<cmath>

/*
// make sure function are inlined to avoid multiple definition
#ifndef __CUDA_ARCH__
#undef __global__
#define __global__ inline __attribute__((always_inline))
#undef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))
#endif
*/

namespace edm {
  template<typename X, typename Y, typename Z>
  __device__ __host__ __forceinline__
  auto fma(X a, Y b, Z c) {
#if defined(EDM_FORCE_FMA) || defined(__FMA__) || defined(FP_FAST_FMA) || defined(__CUDA_ARCH__)
   return std::fma(a,b,c);
#else
   return a*b+c;
#endif
  }
}

template<typename T>
__device__ __host__
T foo(T a, T b) {
  return std::sqrt(a*a-b*b);
}

template<typename T>
__device__ __host__
T foo2(T a, T b) {
  return (a*a-b*b);
}

__device__ __host__
inline
double bar(double a, double b) {
  return std::sqrt(std::fma(a,a,-b*b));
}

__device__ __host__
inline
double edmbar(double a, double b) {
  return std::sqrt(edm::fma(a,a,-b*b));
}

#include<iostream>
#include <cstdlib>
#include<cstdio>


__global__
void doit(double x, double y) {

  auto s = foo(x,y);
  printf("on device:\n%a\n%a\n%a\n%a\n",s,foo2(x,y),bar(x,y),edmbar(x,y));

}

int main(int argc, char** argv) {

#ifdef EDM_FORCE_FMA
    std::cout << "force use of fma" << std::endl;
#endif

#ifdef __FMA__
  std::cout << "hardware fma supported" << std::endl;
#endif

#ifdef FP_FAST_FMA
  std::cout << "fast fma supported" << std::endl;
#endif


  // double x = 0x1.3333333333333p+0;
  double x =  884279719003555.0; // 1.2;
  double y=x;
  if (argc>1) x=atof(argv[1]);
  if (argc>2) y=atof(argv[2]);
  auto s = foo(x,y);
  std::cout << std::hexfloat << s << std::endl;
  std::cout << std::hexfloat << foo2(x,y) << std::endl;
  std::cout << bar(x,y) << std::endl;
  std::cout << edmbar(x,y) << std::endl;


  doit<<<1,1,0,0>>>(x,y);
  cudaDeviceSynchronize();

}

