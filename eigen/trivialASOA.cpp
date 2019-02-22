#include<cstdint>


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


template<uint32_t S>
struct alignas(128) SOA {

  static constexpr uint32_t stride() { return S; }
  static constexpr uint32_t mask() { return S-1;}
  static constexpr uint32_t shift() { return ilog2(S); }
  
  float a[S];
  float b[S];

  static_assert(S>1);
  static_assert(S == 1<<ilog2(S));
  static_assert(sizeof(a)%128 == 0);
};


constexpr uint32_t N = 1024;
constexpr uint32_t S = 256;


using V = SOA<S>;
void sum(V * psoa) {
  #pragma GCC ivdep
  for (uint32_t i=0; i<N; i++) {
    auto j = i>>V::shift();
    auto k = i&V::mask();
    auto & soa = psoa[j];
    soa.b[k] += soa.a[k];
  }
}

void sum0(V * psoa) {
  #pragma GCC ivdep
  for (uint32_t i=0; i<N; i++) {
    auto j = i/S;
    auto k = i%S;
    auto & soa = psoa[j];
    soa.b[k] += soa.a[k];
  }
}

void sum1(V * psoa) {
  #pragma GCC ivdep
  for (uint32_t i=0, j=0, k=0; i<N; i++) {
    auto & soa = psoa[j];
    soa.b[k] += soa.a[k];
    k++;
    if (k==V::stride()) {k=0; j++;}
  }
}


void sum2(V * psoa) {
  auto nb = (N+S-1)/S;
  #pragma GCC ivdep
  for (uint32_t j=0; j<nb; j++) {
    auto & soa = psoa[j];
    for(uint32_t k=0; k<V::stride(); k++) {
     soa.b[k] += soa.a[k];
    }
  }
}
