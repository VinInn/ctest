#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>

// credit    https://marc-b-reynolds.github.io/math/2020/03/11/SinCosPi.html


inline uint32_t f32_to_bits(float x)   { uint32_t u; memcpy(&u,&x,4); return u; }
inline float f32_from_bits(uint32_t x) { float u;    memcpy(&u,&x,4); return u; }

inline float f32_mulsign(float v, uint32_t s) { return f32_from_bits(f32_to_bits(v)^s); }

// constants for sin(pi x) and cos(pi x) for x on [-1/4,1/4]
constexpr float f32_sinpi_7_k[] = { 0x1.921fb6p1f,  -0x1.4abbecp2f, 0x1.466b2p1f,  -0x1.2f5992p-1f };
// constexpr float f32_cospi_6_k[] = { 0x1.fffffep-1f, -0x1.3bd37ep2f, 0x1.03acccp2f, -0x1.4dfd3ap0f  };
// constexpr float f32_cospi_6_k[] = {1.f, -4.93479156494140625f, 4.057690143585205078125f, -1.30715453624725341796875f};
constexpr float f32_cospi_6_k[] = { 0x1p0, -0x1.3bd3ap2, 0x1.03b132p2, -0x1.4ea1aep0};

inline 
void f32_sincospi(float* dst, float a)
{

  auto fmaf = [](float a, float b, float c) { return a*b+c;};

  const float* S = f32_sinpi_7_k;
  const float* C = f32_cospi_6_k;

  float    c,s,a2,a3,r;
  uint32_t q,sx,sy;

  r  = nearbyintf(a+a);
  a  = fmaf(r,-0.5f,a);
  q  = (uint32_t)((int32_t)r);
  a2 = a*a;
  sy = (q<<31); sx = (q>>1)<<31; sy ^= sx; q &= 1; 
  
  c  = fmaf(C[3], a2, C[2]); s = fmaf(S[3], a2, S[2]); a3 = a2*a;
  c  = fmaf(c,    a2, C[1]); s = fmaf(s,    a2, S[1]); 
  c  = fmaf(c,    a2, C[0]); s = a3 * s;
  c  = f32_mulsign(c,sx);    s = fmaf(a,    S[0], s);
                             s = f32_mulsign(s,sy);

  // dst[q  ] = s;
  // dst[q^1] = c;

  dst[0] = q==0 ? s :c;
  dst[1] = q==0 ? c :s;

}


inline float f32_sinpi(float x) {
 float sc[2];
 f32_sincospi(sc,x);
 return sc[0];
}

inline float f32_cospi(float x) {
 float sc[2];
 f32_sincospi(sc,x);
 return sc[1];
}
