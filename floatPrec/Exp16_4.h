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
    double c = (emax/std::numeric_limits<uint16_t>::max());
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

