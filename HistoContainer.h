#include<cstdint>
#include<algorithm>
#ifndef __NVCC__
#include<atomic>
#endif
#ifdef __NVCC__
#include <cuda_runtime.h>
#endif


#ifdef __NVCC__

  template<typename Histo>
  __host__
  void zero(Histo * h, int nthreads, cudaStream_t stream) {
    auto nblocks = (Histo::nbins+nthreads-1)/nthreads;
    zeroOne<<<nblocks,nthreads, 0, stream>>>(h);
  }

  template<typename Histo, typename T>
  __host__
  void fillFromVector(Histo * h, T const * v, uint32_t size, int nthreads, cudaStream_t stream) {
    zero(h,nthreads,stream);
    auto nblocks = (size+nthreads-1)/nthreads;
    fillOneFromVector<<<nblocks,nthreads, 0, stream>>>(h,v,size);
  }


  template<typename Histo>
  __global__
  void zeroOne(Histo * h) {
    h->nspills=0;
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<Histo::nbins) h->n[i]=0;
  }


  template<typename Histo, typename T>
  __global__
  void fillOneFromVector(Histo * h, T const * v, uint32_t size) {
     auto i = blockIdx.x*blockDim.x + threadIdx.x;
     if(i<size) h->fill(v,i);
  }

#endif

template<typename T, uint32_t N, uint32_t M>
class HistoContainer {
public:
  static constexpr uint32_t sizeT = sizeof(T)*8;
  static constexpr uint32_t nbins = 1<<N;
  static constexpr uint32_t shift = sizeT -N;
  static constexpr uint32_t mask =  nbins-1;
  static constexpr uint32_t binSize=1<<M;
  static constexpr uint32_t spillSize=4*binSize;

  static constexpr uint32_t bin(T t) {
    return (t>>shift)&mask;
  }


#ifdef __NVCC__

  __device__
  void fill(T const * t, uint32_t j) {
    auto b = bin(t[j]);
    auto w = atomicAdd(&n[b],1); // atomic
    if (w<binSize) {
      bins[b*binSize+w] = j;
    } else {
      auto w = atomicAdd(&nspills,1); // atomic
      if (w<spillSize) spillBin[w] = j;
    }
  }     

#else
  
  HistoContainer() { zero();}

  void zero() {
   nspills=0;
   for (auto & i : n) i=0;
  }

  void fill(T const * t, uint32_t j) {
    auto b = bin(t[j]);
    auto w = n[b]++; // atomic
    if (w<binSize) {
      bins[b*binSize+w] = j;
    } else {
      auto w = nspills++; // atomic
      if (w<spillSize) spillBin[w] = j;
    }     
  }
#endif

  constexpr bool fullSpill() const {
    return nspills>=spillSize;
  }

  constexpr bool full(uint32_t b) const {
    return n[b]>=binSize;
  }

  constexpr auto const * begin(uint32_t b) const {
     return bins+b*binSize;
  }

  constexpr auto const * end(uint32_t b) const {
     return begin(b)+std::min(binSize,uint32_t(n[b]));
  }

  constexpr auto size(uint32_t b) const {
     return n[b];
  }

#ifdef __NVCC__
  using Counter = uint32_t;
#else
  using Counter = std::atomic<uint32_t>;
#endif

  uint32_t bins[nbins*binSize];
  Counter  n[nbins];
  uint32_t spillBin[spillSize];
  Counter  nspills;

};
