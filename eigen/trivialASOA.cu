#include<cstdint>
#include<algorithm>
#include<tuple>

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


template<typename SOA, int32_t S=SOA::stride()> 
class ASOA {
public:

  static constexpr int32_t stride() { return S; }
  static_assert(isPowerOf2(S),"stride not a power of 2");

  // given a capacity return the required size of the data array
  // given the size will return the number of filled SOAs.
  static constexpr int32_t dataSize(int32_t icapacity) {
     return (icapacity+stride()-1)/stride();
  }

  struct Indices{int32_t j,k;};

  // return the index of the SOA and the index of the element in it
  // in c++17:  auto [j,k] = asoa.indeces(i); auto & soa = asoa[j];  soa.x[k];
  //static constexpr
  //auto indices(int32_t i) { return std::make_tuple(i/stride(), i%stride());}
  // in cuda: auto jk =  asoa.indeces(i); auto & soa = asoa[jk.j];  soa.x[jk.k];
  static constexpr
  auto indices(int32_t i) { return Indices{i/stride(), i%stride()};}

  __device__ __host__
  void construct(int32_t icapacity, SOA * idata) {
    m_size = 0;
    m_capacity = icapacity;
    m_data = idata;
  }


  inline constexpr bool empty() const { return 0 == m_size; }
  inline constexpr bool full() const { return m_capacity == m_size; }
  inline constexpr void clear() { m_size = 0; }
  inline constexpr auto size() const { return m_size; }
  inline constexpr auto capacity() const { return m_capacity; }

  // these manage the SOA themselves
  inline constexpr SOA & operator[](int32_t i) { return m_data[i]; }
  inline constexpr const SOA& operator[](int32_t i) const { return m_data[i]; }
  inline constexpr SOA * data() { return m_data; }
  inline constexpr SOA const * data() const { return m_data; }


  __device__
  int32_t addOne() {
    auto previousSize = atomicAdd(&m_size, 1);
    if (previousSize < m_capacity) {
      return previousSize;
    } else {
      atomicSub(&m_size, 1);
      return -1;
    }
  }

private:
  int32_t m_capacity;
  int32_t m_size=0;

  SOA * m_data;
};


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

using AV = ASOA<V>;

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
void sum(AV * pasoa) {
  auto & asoa = *pasoa;
  int32_t first = threadIdx.x + blockIdx.x*blockDim.x;
  for (auto i=first,n=asoa.size(); i<n; i+=blockDim.x*gridDim.x) {
    auto jk = AV::indices(i);
    auto & soa = asoa[jk.j];
    soa.b[jk.k] += soa.a[jk.k];
  }
}

__global__
void fill(AV * pasoa) {
  auto & asoa = *pasoa;
  auto i = asoa.addOne();
  if (i<0) return;
  auto jk = AV::indices(i);
  auto & soa = asoa[jk.j];
  soa.b[jk.k] = soa.a[jk.k] = threadIdx.x + blockIdx.x*blockDim.x;;
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

