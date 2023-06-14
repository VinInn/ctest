#pragma once

#include "approx_log.h"
#include "sincospi.h"
#include "simpleSinCosPi.h"
#include "luxFloat.h"



namespace fastNormalPDF {


inline std::tuple<float, float> fromFloat(float x, float y) {
  // unsafe_log is very safe here as it cannot return neither Nan nor -inf
  float u = std::sqrt(-2.f * unsafe_logf<6>(x));
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
  return fromFloat(x,y);
}

inline  std::tuple<float, float> from32(uint64_t r) {
  using namespace approx_math;
  constexpr uint64_t mask = std::numeric_limits<uint32_t>::max();
  constexpr float den = 1./(mask+1);
  uint32_t a = r&mask;
  uint32_t b = r>>32;
  float x = den * float(a);
  float y = den * float(b);
  return fromFloat(x,y);
}


inline  std::tuple<float, float> fromMix(uint64_t r) {
  uint32_t v = r & 0x007FFFFF;
  auto y= f32_from_bits((126<<23)|v)-0.75f;  //[-0.25,0.25[
  auto [s,c] =  sincospi0(y);
  uint32_t cs =  r & (1<<23);  // cos sign
  cs  = cs<<8;  // move to position 31
  c = f32_from_bits(cs|f32_to_bits(c));
  uint32_t sw = r &1<<24;  // switch
  r >>= 25;  // 39 bits left
  constexpr float den = 1./(1ULL<<39);
  auto x = den *float(r);
  float u = std::sqrt(-2.f * unsafe_logf<6>(x));
  return {u * ( sw ? c : s) , u * ( sw ? s : c)};
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

// assume G returns 64 bits...
template<typename G>
void genArrayLux(G & gen, float *__restrict__ out, int N) {
  OneIntGen gen32(gen);
  auto one = N%2;
  auto N2 = N+one;
  float r[N2];
  for (int k = 0; k < N2; ++k) r[k] = luxFloat(gen32);
  for (int k = 0; k < N/2; ++k) {
     auto [x, y] =   fromFloat(r[k],r[k+N/2]);
     out[k] = x;
     out[k + N/2] = y;
  }
  if (one) {
    auto [x, y] =   fromFloat(r[N-1],r[N]);
    out[N-1] = x;
  }
}

} // namespace
