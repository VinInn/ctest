#pragma once

#include "Horner.h"
#include <cmath>
#include <limits>
#include <cstdint>

struct LUT16 {
  HD_INLINE LUT16() {}
  template<typename F>
  HD_INLINE LUT16(F f,double emax) {
    double c = (emax/std::numeric_limits<uint16_t>::max());
    for (int i=0; i<65536; i++) {
      lut[i] = f(c*i);
    }
  }

  HD_INLINE float operator[](int i) const { return lut[i];} 
  HD_INLINE float operator()(int i) const { return lut[i];}
  float lut[65536];
};

