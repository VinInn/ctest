#pragma once

#include "Horner.h"
#include <cmath>
#include <limits>
#include <cstdint>

template<int N>
struct LUT {
  static constexpr int NBins = N
  HD_INLINE LUT16() {}
  template<typename F>
  HD_INLINE LUT(F f,double emax) {
    double c = (emax/NBINS);
    for (int i=0; i<NBins; i++) {
      lut[i] = f(c*i);
    }
  }

  HD_INLINE float operator[](int i) const { return lut[i];} 
  HD_INLINE float operator()(int i) const { return lut[i];}
  float lut[NBins];
};

using LUT16 = LUT<65536>;
