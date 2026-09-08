#pragma once

#include "Horner.h"
#include <cmath>
#include <limits>
#include <cstdint>
#include <bit>

template<int N, uint32_t MAX>
struct LUT {
  static constexpr int NBins = 1<<N;
  static constexpr float fmax = std::bit_cast<float>(MAX);
  static constexpr float coeff = fmax/float(NBins);
  static constexpr float coefi = float(NBins)/fmax;
  static HD_INLINE float tof(int i) { return coeff*i; }
  static HD_INLINE float toi(float x) { return std::round(coefi*x); }

  HD_INLINE LUT() {}
  template<typename F>
  HD_INLINE explicit LUT(F f) {
    for (int i=0; i<NBins; i++) {
      lut[i] = f(tof(i));
    }
  }

  HD_INLINE float operator[](int i) const { return lut[i];} 
  HD_INLINE float operator()(int i) const { return lut[i];}
  float lut[NBins];
};


template<uint32_t MAX>
using LUT16 = LUT<16,MAX>;
