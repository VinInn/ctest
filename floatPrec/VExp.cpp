/*
 * compile with
 * c++ -std=c++1y -Ofast VExp.cpp -Wall -Wno-psabi -S -fabi-version=0  -DNOMAIN
 * c++ -std=c++1y -Ofast VExp.cpp -Wall -Wno-psabi -S -fabi-version=0 -march=native
 * use -DDOFMA for targets supporting fma  (can be automatized...)
 *
   c++ -std=c++1y -Ofast VExp.cpp -Wall -Wno-psabi -S -fabi-version=0 -DNOMAIN -msse3; cat VExp.s
   c++ -std=c++1y -Ofast VExp.cpp -Wall -Wno-psabi -S -fabi-version=0 -DNOMAIN -march=nehalem; cat VExp.s
   c++ -std=c++1y -Ofast VExp.cpp -Wall -Wno-psabi -S -fabi-version=0 -DNOMAIN -march=bdver1  -DDOFMA; cat VExp.s
   c++ -std=c++1y -Ofast VExp.cpp -Wall -Wno-psabi -S -fabi-version=0 -DNOMAIN -march=sandybridge; cat VExp.s
   c++ -std=c++1y -Ofast VExp.cpp -Wall -Wno-psabi -S -fabi-version=0 -DNOMAIN -march=haswell  -DDOFMA; cat VExp.s
   c++ -std=c++1y -Ofast VExp.cpp -Wall -Wno-psabi -S -fabi-version=0 -DNOMAIN -mavx512f -mfma  -DDOFMA; cat VExp.s 
 */

#include "approx_vexp.h"


#include<cstdio>
#include<cstdlib>
#include<iostream>

typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;
typedef float __attribute__( ( vector_size( 32 ) ) ) float32x8_t;
typedef float __attribute__( ( vector_size( 64 ) ) ) float32x16_t;

// namespace {
#ifdef DOFMA
#define fma true
// constexpr bool fma=true;
#else
#define fma false
  // constexpr bool fma=false;
#endif
//}


float
 myExp(float vx) {
  return approx_expf<decltype(vx),6,fma>(vx);
}


float32x4_t
 myExp(float32x4_t vx) {
  return approx_expf<decltype(vx),6,fma>(vx);
}


float32x8_t 
myExp(float32x8_t vx) {
  return approx_expf<decltype(vx),6,fma>(vx);
}

float32x16_t 
myExp(float32x16_t vx)  {
  return approx_expf<decltype(vx),6,fma>(vx);
}

#ifndef NOMAIN

int main() {


  float x = 0.15;
  auto y1 = approx_expf<float,6,fma>(x);
  auto y2 = approx_expf<float,6,fma>(-x);

  std::cout << x << ' ' << y1 << ' ' << y2 << std::endl;


  float32x4_t vx = { 0.15, -0.15, 2.1, -2.1};

  auto yv = approx_expf<decltype(vx),6,fma>(vx);
  
  std::cout << yv[0] << ' '  << yv[1] << ' ' << yv[2] << ' ' << yv[3] << std::endl;

  return 0;

}


#endif
