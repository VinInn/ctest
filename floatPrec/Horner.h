#pragma once
#include<cmath>
template<int N>
float horner(float x, float const * const c) {
#if  defined(__FMA__) || defined(FP_FAST_FMA)
  return std::fma(x,horner<N-1>(x,c),c[N]);
#else
  return x*horner<N-1>(x,c)+c[N];
#endif
}

template<>
inline
float horner<0>(float x, float const * const c) {
  return c[0];
}
