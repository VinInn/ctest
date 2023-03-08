#pragma once

#include "approx_log.h"
#include "sincospi.h"
#include "luxFloat.h"



namespace fastNormalPDF {


inline std::tuple<float, float> fromFloat(float x, float y) {
  // unsafe_log is very safe here as it cannot return neither Nan nor -inf
  float u = std::sqrt(-2.f * unsafe_logf<8>(x));
  auto [s, c] = f32_sincospi(2.f * y);
  return {u * c, u * s};
}


// generate two normal distributed single precision number (mean 0, variance 1) from 64 random bits
inline std::tuple<float, float> from23(uint64_t r) {
  using namespace approx_math;
  binary32 fi;
  fi.ui32 = r & 0x007FFFFF;
  fi.ui32 |= 0x3F800000;  // extract mantissa as an FP number
  float xmin = 1./(0x007FFFFF+1);
  auto x = xmin +(fi.f - 1.f);
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

inline  std::tuple<float, float> from32(uint64_t r) {
  using namespace approx_math;
  constexpr uint64_t mask = std::numeric_limits<uint32_t>::max();
  constexpr float den = 1./(mask+1);
  uint32_t a = r&mask;
  uint32_t b = r>>32;
  float x = den * float(a);
  float y = den * float(b);
  // unsafe_log is very safe here as it cannot return neither Nan nor -inf
  float u = std::sqrt(-2.f * unsafe_logf<8>(x+den));
  auto [s, c] = f32_sincospi(2.f * y);
  return {u * c, u * s};
}

inline  std::tuple<float, float> fromMix(uint64_t r) {
  using namespace approx_math;
  binary32 fi;
  fi.ui32 = r & 0x007FFFFF;
  fi.ui32 |= 0x3F800000;  // extract mantissa as an FP number
  auto y = fi.f - 1.f;
  r >>= 23;  // 41 bits left
  constexpr float den = 1./(1ULL<<41);
  auto x = den *float(r);
  // unsafe_log is very safe here as it cannot return neither Nan nor -inf
  float u = std::sqrt(-2.f * unsafe_logf<8>(x));
  auto [s, c] = f32_sincospi(2.f * y);
  return {u * c, u * s};
}


// assume G returns 64 bits...
template<typename F, typename G>
void genArray(F f, G & gen, float *__restrict__ out, int N) {
  auto one = N%2;
  N /= 2;
  uint64_t r[N];
  for (int k = 0; k < N; ++k) r[k] = gen(); 
  for (int k = 0; k < N; ++k) {
      auto [x, y] = f(r[k]);
      out[k] = x;
      out[k + N] = y;
  }
  if (one) {
    auto [x, y] = f(gen());
    out[N + N] = x;
  }
}

} // namespace
