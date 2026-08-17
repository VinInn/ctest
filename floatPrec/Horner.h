#pragma once
#include<cmath>
template<int N>
float horner(float x, float const * const c) {
  return std::fma(x,horner<N-1>(x,c),c[N]);
}

template<>
inline
float horner<0>(float x, float const * const c) {
  return c[0];
}
