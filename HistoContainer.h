#include<cstdint>
#include<algorithm>
#include<atomic>

template<typename T, uint32_t N, uint32_t M>
class HistoContainer {

  static constexpr uint32_t sizeT = sizeof(T)*4;
  static constexpr uint32_t nbins = 1<<N;
  static constexpr uint32_t shift = sizeT -N;
  static constexpr uint32_t mask =  nbins-1;
  static constexpr uint32_t binSize=M;
  static constexpr uint32_t spillSize=4*binSize;

  static constexpr uint32_t bin(T t) {
    return (t>>shift)&mask;
  }

#ifdef NVCC

#else
  
  HistoContainer() { zero();}

  void zero() {
   nspills=0;
   for (auto & i : n) i=0;
  }

  void fill(T t) {
    auto b = bin(t);
    auto w = n[b]++; // atomic
    if (w<binSize) {
      bins[b*binSize+w] =t;
    } else {
      auto w = nspills++; // atomic
      if (w<spillSize) spillBin[w]=t;
    }     
  }
#endif

  constexpr bool fullSpill() const {
    return nspills>=spillSize;
  }

  constexpr bool full(uint32_t b) const {
    return n[b]>=binSize;
  }

  constexpr T const * begin(uint32_t b) const {
     return bins+b*binSize;
  }

  constexpr T const * end(uint32_t b) const {
     return begin(b)+std::min(binSize,n[b]);
  }

  constexpr uint32_t size(uint32_t b) const {
     return n[b];
  }

#ifdef NVCC
  using Counter = uint32_t;
#else
  using Counter = std::atomic<uint32_t>;
#endif

  T bins[nbins*binSize];
  Counter  n[nbins];
  T spillBin[spillSize];
  Counter  nspills;

};
