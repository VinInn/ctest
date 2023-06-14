#pragma once

#include "XoshiroSOA.h"


/* This is xoshiro256++ and xoshiro256** 1.0, two of Blackman&Vigna all-purpose, rock-solid generators.
   It has excellent (sub-ns) speed, a state (256 bits) that is large
   enough for any parallel application, and it passes all tests the authors are
   aware of.

   For generating just floating-point numbers, xoshiro256+ is even faster.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s. */

#ifdef __AVX2__
typedef uint64_t XoshiroVector __attribute__ ((vector_size (32)));
#else
typedef uint64_t XoshiroVector __attribute__ ((vector_size (16)));
#endif

template <XoshiroType type, typename V> class Xoshiro;

// xoshiro256++
using XoshiroPP = Xoshiro<XoshiroType::TwoSums,XoshiroVector>;
// xoshiro256**
using XoshiroSS = Xoshiro<XoshiroType::TwoMuls,XoshiroVector>;
// xoshiro256+
using XoshiroP = Xoshiro<XoshiroType::OneSum,XoshiroVector>;

template <XoshiroType type, typename V=XoshiroVector> 
class Xoshiro {
public:

  using vector_type = V;
  static constexpr int32_t vector_size = sizeof(vector_type)/sizeof(uint64_t);
  using store_type = vector_type;
  using result_type = uint64_t;
  static constexpr uint64_t min() { return  std::numeric_limits<uint64_t>::min(); }
  static constexpr uint64_t max() { return  std::numeric_limits<uint64_t>::max(); }

  explicit Xoshiro(uint64_t seed=0) {
    using namespace xoshiroRNG;
    SplitMix64 g(seed);
    if constexpr (1==vector_size) for ( auto & s : m_s) s=g();
    else {
       uint64_t ls[4]; for ( auto & s : ls) s=g();
       for (int i=0; i<4; ++i) m_s[i][0] = ls[i];
       for (int j=1; j<vector_size; ++j) {
         jump(ls);
         for (int i=0; i<4; ++i) m_s[i][j] = ls[i];
       }
    }
  }

  uint64_t operator()() {
    if constexpr (1==vector_size) return next();
    else {
      if (vector_size==m_n) {
        m_res = next();
        m_n=0;
      }
      return m_res[m_n++];
    }
  }

  vector_type next() {
    if constexpr (type==XoshiroType::TwoSums) return nextPP();
    if constexpr (type==XoshiroType::TwoMuls) return nextSS();
    if constexpr (type==XoshiroType::OneSum) return nextP();
  }

private: 

  vector_type m_s[4];

  store_type m_res;
  int m_n=vector_size;

public:
  // xoshiro256**
  auto nextSS() {
    using namespace xoshiroRNG;
    const auto result = rotl(m_s[1] * 5, 7) * 9;
    advance(m_s);
    return result;
  }

  // xoshiro256++
  auto nextPP() {
    using namespace xoshiroRNG;
    const auto result = rotl(m_s[0] + m_s[3], 23) + m_s[0];
    advance(m_s);
    return result;
  }

  // xoshiro256+  fastest generator for floating-point numbers by extracting the upper 53 bits
  auto nextP() {
    using namespace xoshiroRNG;
    const auto result = m_s[0] + m_s[3];
    advance(m_s);
    return result;
  }


};
