#pragma once

/*  Written in 2019 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

// C++ Version by Vincenzo Innocente 2023


#include <cstdint>
#include <limits>

/* This is a fixed-increment version of Java 8's SplittableRandom generator
   See http://dx.doi.org/10.1145/2714064.2660195 and
   http://docs.oracle.com/javase/8/docs/api/java/util/SplittableRandom.html

   It is a very fast generator passing BigCrush, and it can be useful if
   for some reason you absolutely want 64 bits of state. */
class SplitMix64 {
private:
  uint64_t x; /* The state can be seeded with any value. */

public:
  using result_type = uint64_t;
  static constexpr uint64_t min() { return  std::numeric_limits<uint64_t>::min(); }
  static constexpr uint64_t max() { return  std::numeric_limits<uint64_t>::max(); }

  explicit SplitMix64(uint64_t seed) : x(seed) {}

  uint64_t operator()() {return next();}

  uint64_t next() {
    uint64_t z = (x += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
  }
};

enum class XoshiroType { TwoSums, TwoMuls, OneSum};

namespace xoshiroRNG {

  template<typename T>
  static constexpr T rotl(const T x, int k) {
    return (x << k) | (x >> (64 - k));
  }

  template<typename T>
  static constexpr void advance(T * s) {

    const auto t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;

    s[3] = rotl(s[3], 45);
  }

  /* This is the jump function for the generator. It is equivalent
   to 2^128 calls to next(); it can be used to generate 2^128
   non-overlapping subsequences for parallel computations. */
  template<typename T>
  inline constexpr void jump(T * s) {
    constexpr uint64_t JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };
    T s0 = 0;
    T s1 = 0;
    T s2 = 0;
    T s3 = 0;
    for(uint32_t i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
        for(int b = 0; b < 64; b++) {
            if (JUMP[i] & UINT64_C(1) << b) {
                s0 ^= s[0];
                s1 ^= s[1];
                s2 ^= s[2];
                s3 ^= s[3];
            }
            advance(s);
        }

    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
  }

  /* This is the long-jump function for the generator. It is equivalent to
   2^192 calls to next(); it can be used to generate 2^64 starting points,
   from each of which jump() will generate 2^64 non-overlapping
   subsequences for parallel distributed computations. */
   template<typename T>
   inline constexpr void long_jump(T * s) {
    constexpr uint64_t LONG_JUMP[] = { 0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635 };
    T s0 = 0;
    T s1 = 0;
    T s2 = 0;
    T s3 = 0;
    for(uint32_t i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
        for(int b = 0; b < 64; b++) {
            if (LONG_JUMP[i] & UINT64_C(1) << b) {
                s0 ^= s[0];
                s1 ^= s[1];
                s2 ^= s[2];
                s3 ^= s[3];
            }
            advance(s);
        }

    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
  }


template<int N>
struct SOA {
  constexpr uint64_t * operator[](int i) { return v[i];}
  constexpr uint64_t const * operator[](int i) const { return v[i];}
  uint64_t* v[4];
  static constexpr int size = N;
};


template<typename SOA>
constexpr void advance(SOA & s, int i) {

    const auto t = s[1][i] << 17;

    s[2][i] ^= s[0][i];
    s[3][i] ^= s[1][i];
    s[1][i] ^= s[2][i];
    s[0][i] ^= s[3][i];

    s[2][i] ^= t;

    s[3][i] = rotl(s[3][i], 45);
}

template<typename SOA>
constexpr auto nextSS(SOA & s, int i) {
    const auto result = rotl(s[1][i] * 5, 7) * 9;
    advance(s,i);
    return result;
}

template<typename SOA>
constexpr auto nextPP(SOA & s, int i) {
    const auto result = rotl(s[0][i] + s[3][i], 23) + s[0][i];
    advance(s,i);
    return result;
}

template<typename SOA>
constexpr auto nextP(SOA & s, int i) {
    const auto result = s[0][i] + s[3][i];
    advance(s,i);
    return result;
  }

template<XoshiroType type, typename SOA>
constexpr auto next(SOA & s, int i) {
   if constexpr (type==XoshiroType::TwoSums) return nextPP(s,i);
   if constexpr (type==XoshiroType::TwoMuls) return nextSS(s,i);
   if constexpr (type==XoshiroType::OneSum) return nextP(s,i);
}

template<typename SOA>
void setSeed(SOA & s, uint64_t seed=0) {
 SplitMix64 g(seed);
 uint64_t ls[4]; for ( auto & s : ls) s=g();
 for (int i=0; i<4; ++i) s[i][0] = ls[i];
 for (int j=1; j<s.size; ++j) {
    jump(ls);
    for (int i=0; i<4; ++i) s[i][j] = ls[i];
  }
}

template<XoshiroType RNG, typename SOA, typename F>
void loop(SOA soa, F & f, int n) {
  auto ni = n/soa.size;
  int j=0;
  for(int i=0; i<ni; ++i) {
     for(int k=0; k<soa.size; ++k) f(j++, next<RNG>(soa,k));
  }
  ni = n - ni*soa.size;
  for(int k=0; k<ni; ++k) f(j++, next<RNG>(soa,k));
}


}

/* example of use in gpu
__global__
 void gen(SOA s, uint64_t * a, int n) {
     // assumption: SOA size < blockDim.x*GridDim.x; (aka the total number of threads)
     // this should ensure that only one thread is updating a given state at any time
     int tid = blockDim.x * blockIdx.x + threadIdx.x;
     for (int i = tid; i < n; i += blockDim.x) {
       a[i] = nextPP(s,tid);
}
*/

