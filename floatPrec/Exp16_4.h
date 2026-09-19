#pragma once 

#include "Horner.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <cstdint>

struct Exp16_4 {

  struct HB {
    uint8_t b0:4;
    uint8_t b1:4;
    uint8_t b2:4;
    uint8_t b3:4;
  };
  union I16 {
    uint16_t i16;
    HB i4;
  };

//#ifdef __CUDA__ARCH__
  HD_INLINE Exp16_4() {}
//#endif 
  HD_INLINE explicit Exp16_4(double emax) {
    double c = ldexp(emax,-16);
    double ce[4] = {std::exp(c),std::exp(ldexp(c,4)),std::exp(ldexp(c,8)),std::exp(ldexp(c,12))};
    for (int i=0; i<16; ++i) {
      for (int j=0; j<4; ++j) {
        pefact[j][i]= std::pow(ce[j],i);
//      std::cout << efact[j][i] << ' ';
      }
//   std::cout << std::endl;
    }
    double nce[4] = {std::exp(-c),std::exp(ldexp(-c,4)),std::exp(ldexp(-c,8)),std::exp(ldexp(-c,12))};
    for (int i=0; i<16; ++i) {
      for (int j=0; j<4; ++j) {
        nefact[j][i]= std::pow(nce[j],i);
//      std::cout << nefact[j][i] << ' ';
      }
//    std::cout << std::endl;
    }
  }

  static HD_INLINE float lut(float const * p, uint8_t j) {
#if defined(__CUDA_ARCH__) & defined(LUT_SHFL)
    auto yy = p[threadIdx.x%16];
    return __shfl_sync(0xFFFFFFFF, yy, j, 16);
#else
    return p[j];
#endif
}

  HD_INLINE float pexp(uint16_t x) const {
    I16 u; u.i16 = x;
// #ifdef LUT_SHFL
    return lut(pefact[0],u.i4.b0)*lut(pefact[1],u.i4.b1)*lut(pefact[2],u.i4.b2)*lut(pefact[3],u.i4.b3);
// #else
//    return (pefact[0][u.i4.b0]*pefact[1][u.i4.b1])*(pefact[2][u.i4.b2]*pefact[3][u.i4.b3]);
// #endif
  }
  HD_INLINE float nexp(uint16_t x) const {
    I16 u; u.i16 = x;
// #ifdef LUT_SHFL
    return lut(nefact[0],u.i4.b0)*lut(nefact[1],u.i4.b1)*lut(nefact[2],u.i4.b2)*lut(nefact[3],u.i4.b3);
// #else
//    return (nefact[0][u.i4.b0]*nefact[1][u.i4.b1])*(nefact[2][u.i4.b2]*nefact[3][u.i4.b3]);
// #endif
  }


float pefact[4][16];
float nefact[4][16];

};

// https://godbolt.org/z/ee4d1nEsx
typedef float __attribute__( ( vector_size( 4*16 ) ) ) float32x16_t;
typedef int   __attribute__( ( vector_size( 4*16 ) ) ) int32x16_t;
typedef uint16_t   __attribute__( ( vector_size( 2*16 ) ) ) int16x16_t;


struct Exp16V {

  explicit ExpV(double emax) {
    double c = ldexp(emax,-16);
    double ce[4] = {std::exp(c),std::exp(ldexp(c,4)),std::exp(ldexp(c,8)),std::exp(ldexp(c,12))};
    for (int i=0; i<16; ++i) {
      for (int j=0; j<4; ++j) {
        pefact[j][i]= std::pow(ce[j],i);
//      std::cout << efact[j][i] << ' ';
      }
//   std::cout << std::endl;
    }
    double nce[4] = {std::exp(-c),std::exp(ldexp(-c,4)),std::exp(ldexp(-c,8)),std::exp(ldexp(-c,12))};
    for (int i=0; i<16; ++i) {
      for (int j=0; j<4; ++j) {
        nefact[j][i]= std::pow(nce[j],i);
//      std::cout << nefact[j][i] << ' ';
      }
//    std::cout << std::endl;
    }
  }


  float32x16_t lut(float32x16_t c, int16x16_t x, int s) {
    int16x16_t b0 = (x>>s)&15;
    int32x16_t m = {b0[0],b0[1],b0[2],b0[3],b0[4],b0[5],b0[6],b0[7],
                    b0[8+0],b0[8+1],b0[8+2],b0[8+3],b0[8+4],b0[8+5],b0[8+6],b0[8+7]};
    return  __builtin_shuffle(c,m);
  }
  float32x16_t pexp(int16x16_t x)  {
    return lut(pefact[0],x,0)*lut(pefact[1],x,4)*lut(pefact[2],x,8)*lut(pefact[3],x,12);
  }
  float32x16_t nexp(int16x16_t x)  {
    return lut(nefact[0],x,0)*lut(nefact[1],x,4)*lut(nefact[2],x,8)*lut(nefact[3],x,12);
  }


  float32x16_t pefact[4];
  float32x16_t nefact[4];

};
