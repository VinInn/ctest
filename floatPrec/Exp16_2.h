#pragma once 

#include "Horner.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <cstdint>

struct Exp16_2 {

  struct HB {
    uint8_t b0;
    uint8_t b1;
  };
  union I16 {
    uint16_t i16;
    HB i8;
  };

// #ifdef __CUDA__ARCH__
  HD_INLINE Exp16_2() {}
// #endif 
  HD_INLINE explicit Exp16_2(double emax) {
    double c = (emax/std::numeric_limits<uint16_t>::max());
    double ce[2] = {std::exp(c),std::exp(ldexp(c,8))};
    for (int i=0; i<256; ++i) {
      for (int j=0; j<2; ++j) {
        pefact[j][i]= std::pow(ce[j],i);
//      std::cout << efact[j][i] << ' ';
      }
//   std::cout << std::endl;
    }
    double nce[2] = {std::exp(-c),std::exp(ldexp(-c,8))};
    for (int i=0; i<256; ++i) {
      for (int j=0; j<2; ++j) {
        nefact[j][i]= std::pow(nce[j],i);
//      std::cout << nefact[j][i] << ' ';
      }
//    std::cout << std::endl;
    }
  }

  static HD_INLINE float lut(float const * p, uint8_t j) {
    return p[j];
}

  HD_INLINE float pexp(uint16_t x) const {
    I16 u; u.i16 = x;
//    return lut(pefact[0],u.i8.b0)*lut(pefact[1],u.i8.b1);
    return (pefact[0][u.i8.b0]*pefact[1][u.i8.b1]);
  }
  HD_INLINE float nexp(uint16_t x) const {
    I16 u; u.i16 = x;
//    return lut(nefact[0],u.i8.b0)*lut(nefact[1],u.i8.b1);
    return (nefact[0][u.i8.b0]*nefact[1][u.i8.b1]);
  }


float pefact[4][256];
float nefact[4][256];

};

