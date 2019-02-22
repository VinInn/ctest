#include<cstdint>
#include<algorithm>

constexpr uint32_t ilog2(uint32_t v) {

  constexpr uint32_t b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
  constexpr uint32_t S[] = {1, 2, 4, 8, 16};

  uint32_t r = 0; // result of log2(v) will go here
  for (auto i = 4; i >= 0; i--)
  if (v & b[i]) {
    v >>= S[i];
    r |= S[i];
  }
  return r;
}

constexpr bool isPowerOf2(uint32_t v) {
    return v && !(v & (v - 1));
}


template<uint32_t S>
struct alignas(128) SOA {

  static constexpr uint32_t stride() { return S; }
  static constexpr uint32_t mask() { return S-1;}
  static constexpr uint32_t shift() { return ilog2(S); }
  
  float a[S];
  float b[S];

  static_assert(isPowerOf2(S),"stride not a power of 2");
  static_assert(sizeof(a)%128 == 0,"size not a multiple of 128");
};

constexpr uint32_t S = 256;


using V = SOA<S>;

__global__
void sum(V * psoa, int n) {
  auto first = threadIdx.x + blockIdx.x*blockDim.x;
  for (auto i=first; i<n; i+=blockDim.x*gridDim.x) {
    auto j = i/V::stride();
    auto k = i%V::stride();
    auto & soa = psoa[j];
    soa.b[k] += soa.a[k];
  }
}


__global__
void sum2(V * psoa, int n) {
  auto nb = (n+V::stride()-1)/V::stride();
  for (auto j=blockIdx.x; j<nb; j+=gridDim.x) {
    auto & soa = psoa[j];
    auto kmax = std::min(V::stride(),n - j*V::stride());
    for(uint32_t k=threadIdx.x; k<kmax; k+=blockDim.x) {
     soa.b[k] += soa.a[k];
    }
  }
}

