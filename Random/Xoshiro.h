/*  Written in 2019 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

// C++ Version by Vincenzo Innocente 2023

#include <cstdint>
#include <limits>
#include <array>


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


/* This is xoshiro256++ and xoshiro256** 1.0, two of Blackman&Vigna all-purpose, rock-solid generators.
   It has excellent (sub-ns) speed, a state (256 bits) that is large
   enough for any parallel application, and it passes all tests the authors are
   aware of.

   For generating just floating-point numbers, xoshiro256+ is even faster.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s. */


typedef uint64_t XoshiroVector __attribute__ ((vector_size (32)));


enum class XoshiroType { TwoSums, TwoMuls, OneSum};

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
    SplitMix64 g(seed);
    for ( auto & s : m_s) {
      if constexpr (1==vector_size) s=g();
      else for (int i=0; i<vector_size; ++i) s[i]=g();
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
/*
  // gpu interface assume m_s is a SOA 
  uint64_t operator()(int i) {
    uint64_t s[4];
    for (int j=0; j<4; ++j) s[j] = m_s[j][i];
    if constexpr (type==XoshiroType::TwoSums) return nextPP(s);
    if constexpr (type==XoshiroType::TwoMuls) return nextSS(s);
    if constexpr (type==XoshiroType::OneSum) return nextP(s);
  }
*/

  vector_type next() {
    if constexpr (type==XoshiroType::TwoSums) return nextPP(m_s);
    if constexpr (type==XoshiroType::TwoMuls) return nextSS(m_s);
    if constexpr (type==XoshiroType::OneSum) return nextP(m_s);
  }

private: 

  vector_type m_s[4];  // for cuda replace with home made struct...

  store_type m_res;
  int m_n=vector_size;

  template<typename T>
  static constexpr T rotl(const T x, int k) {
    return (x << k) | (x >> (64 - k));
  }

  template<typename T>
  static constexpr void advance(T* s) {
    const auto t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;

    s[3] = rotl(s[3], 45);
  } 

public:
  // xoshiro256**
  template<typename T>
  static constexpr T nextSS(T*s) {
    const auto result = rotl(s[1] * 5, 7) * 9;
    advance(s);
    return result;
  }

  // xoshiro256++
  template<typename T>
  static constexpr T nextPP(T*s) {
    const auto result = rotl(s[0] + s[3], 23) + s[0];
    advance(s);
    return result;
  }

  // xoshiro256+  fastest generator for floating-point numbers by extracting the upper 53 bits
  template<typename T>
  static constexpr T nextP(T*s) {
    const T result = s[0] + s[3];
    advance(s);
    return result;
  }


   static constexpr uint64_t JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };
   static constexpr uint64_t LONG_JUMP[] = { 0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635 };

  /* This is the jump function for the generator. It is equivalent
   to 2^128 calls to next(); it can be used to generate 2^128
   non-overlapping subsequences for parallel computations. */
  template<typename T> 
  static constexpr void jump(T*s) {

    T s0 = 0;
    T s1 = 0;
    T s2 = 0;
    T s3 = 0;
    for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
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
  static constexpr void long_jump(T*s) {

    T s0 = 0;
    T s1 = 0;
    T s2 = 0;
    T s3 = 0;
    for(int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
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


};
