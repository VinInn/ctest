#pragma once

#include "Horner.h"
#include <cmath>
#include <limits>
#include <cstdint>

template<int N>
struct LUT {
  // static constexpr int NBits = N;
  static constexpr int NBins = 1<<N;
  HD_INLINE LUT() {}
  template<typename F>
  HD_INLINE LUT(F f,double emax) {
    double c = (emax/(NBins));
    for (int i=0; i<NBins; i++) {
      lut[i] = f(c*i);
    }
  }

  HD_INLINE float operator[](int i) const { return lut[i];} 
  HD_INLINE float operator()(int i) const { return lut[i];}
  float lut[NBins];
};

using LUT16 = LUT<16>;
