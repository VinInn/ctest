#pragma once

#include "approx_log.h"
#include "sincospi.h"

// generate two normal distributed single precision number (mean 0, variance 1) from 64 random bits
inline std::tuple<float, float> fastNormalPDF(uint64_t r) {
  using namespace approx_math;
  binary32 fi;
  fi.ui32 = r & 0x007FFFFF;
  fi.ui32 |= 0x3F800000;  // extract mantissa as an FP number
  auto x = fi.f - 1.f;
  r >>=23;  // discard those used
  // repeat
  fi.ui32 = r & 0x007FFFFF;
  fi.ui32 |= 0x3F800000;  // extract mantissa as an FP number
  auto y = fi.f - 1.f;
  // unsafe_log is very safe here as it cannot return neither Nan nor -inf
  float u = std::sqrt(-2.f * unsafe_logf<8>(x));
  auto [s, c] = f32_sincospi(2.f * y);
  return {u * c, u * s};
}

inline 
void fastNormalPDF(uint64_t const *__restrict__ r, float *__restrict__ out, int N) {
    for (int k = 0; k < N; ++k) {
      auto [x, y] = fastNormalPDF(r[k]);
      out[k] = x;
      out[k + N] = y;
    }
}
