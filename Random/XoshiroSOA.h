#include <cstdint>
#include<array>

/* This is xoshiro256++ 1.0, one of our all-purpose, rock-solid generators.
   It has excellent (sub-ns) speed, a state (256 bits) that is large
   enough for any parallel application, and it passes all tests we are
   aware of.

   For generating just floating-point numbers, xoshiro256+ is even faster.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s. */

namespace xoshiroSOA {

template<typename V>
constexpr V rotl(const V x, int k) {
    return (x << k) | (x >> (64 - k));
}

template< typename T>
constexpr void advance(T & s, int i) {

    const auto t = s[1][i] << 17;

    s[2][i] ^= s[0][i];
    s[3][i] ^= s[1][i];
    s[1][i] ^= s[2][i];
    s[0][i] ^= s[3][i];

    s[2][i] ^= t;

    s[3][i] = rotl(s[3][i], 45);
}

template< typename T>
constexpr auto nextSS(T & s, int i) {
    const auto result = rotl(s[1][j] * 5, 7) * 9;
    advance(s,i);
    return result;
}

template< typename T>
constexpr auto nextPP(T & s, int i) {
    const auto result = rotl(s[0][i] + s[3][i], 23) + s[0][i];
    advance(s,i);
    return result;
}

template< typename T>
constexpr auto nextP(T & s, int i) {
    const auto result = s[0][i] + s[3][i];
    advance(s,i);
    return result;
  }


template<typename T,typename G>
setSeed(T&s, G & g) {
 for (int j=0; j<4; ++j)
 for (int i=0; i<s.size; ++i) 
   s[j][i] = g();
}


struct SOA {
  constexpr uint64_t * operator[](int i) { return v[i];}    
  constexpr uint64_t const * operator[](int i) const { return v[i];}
  uint64_t* v[4];
  int size;
};

}

/* example of use
__global__
 void gen(SOA s, uint64_t * a, int n) {
     // assumption: SOA size < blockDim.x*GridDim.x; (aka the total number of threads)
     // this should ensure that only one thread is updating a given state at any time
     int tid = blockDim.x * blockIdx.x + threadIdx.x;
     for (int i = tid; i < n; i += blockDim.x) {
       a[i] = nextPP(s,tid);
}
*/

