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
//#ifndef __CUDA_ARCH__
  template<typename F>
  HD_INLINE explicit LUT(F f) {
    for (int i=0; i<NBins; i++) {
      if constexpr(MAX>0) {
        lut[i] = f(tof(i));
      } else {
        lut[i] = f(i);
      }
    }
  }
//#endif

#ifdef __CUDA_ARCH__
  HD_INLINE float operator[](int i) const { return __ldg(lut+i);} 
  HD_INLINE float operator()(int i) const { return __ldg(lut+i);}
#else
  HD_INLINE float operator[](int i) const { return lut[i];}
  HD_INLINE float operator()(int i) const { return lut[i];}
#endif
  float lut[NBins];
};


template<int N, uint32_t XMAX,  uint32_t YMAX>
struct LUT16 {
  static constexpr int NBins = 1<<N;
  static constexpr int max16 = 65536;
  static constexpr float fmax = std::bit_cast<float>(XMAX);
  static constexpr float coeff = fmax/float(NBins);
  static constexpr float coefi = float(NBins)/fmax;
  static HD_INLINE float tof(int i) { return coeff*float(i); }
  static HD_INLINE float toi(float x) { return std::round(coefi*x); }

  static constexpr float ymax = std::bit_cast<float>(YMAX);
  static constexpr float ycoeff = ymax/float(max16);
  static constexpr float ycoefi = float(max16)/ymax;
  static HD_INLINE float ytof(int i) { return ycoeff*float(i); }
  static HD_INLINE float ytoi(float x) { return std::round(ycoefi*x); }


  HD_INLINE LUT16() {}
// #ifndef __CUDA_ARCH__
  template<typename F>
  HD_INLINE explicit LUT16(F f) {
    for (int i=0; i<NBins; i++) {
      if constexpr(XMAX>0) {
        lut[i] = ytoi(f(tof(i)));
      } else {
        lut[i] = f(i);
      }
    }
  }
// #endif

#if defined __CUDA_ARCH__  && TEXTURE
  HD_INLINE float operator[](int i) const { return ytof(__ldg(lut+i));}
  HD_INLINE float operator()(int i) const { return ytof(__ldg(lut+i));}
#else
  HD_INLINE float operator[](int i) const { return ytof(lut[i]);}
  HD_INLINE float operator()(int i) const { return ytof(lut[i]);}
#endif
  uint16_t lut[NBins];
};
