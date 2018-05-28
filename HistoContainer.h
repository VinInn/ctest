#ifndef HeterogeneousCoreCUDAUtilitiesHistoContainer_h
#define HeterogeneousCoreCUDAUtilitiesHistoContainer_h


#include<cassert>
#include<cstdint>
#include<algorithm>
#ifndef __NVCC__
#include<atomic>
#endif
#ifdef __NVCC__
#include <cuda_runtime.h>
#endif


#ifdef __NVCC__

  template<class ForwardIt, class T>
  __device__
  constexpr
  ForwardIt upper_bound(ForwardIt first, ForwardIt last, const T& value)
  {
    auto count = last-first;
 
    while (count > 0) {
        auto it = first; 
        auto step = count / 2; 
        it+=step;
        if (!(value < *it)) {
            first = ++it;
            count -= step + 1;
        } 
        else
            count = step;
    }
    return first;
  }


  template<typename Histo>
  __host__
  void zero(Histo * h, uint32_t nh, int nthreads, cudaStream_t stream) {
    auto nblocks = (nh*Histo::nbins+nthreads-1)/nthreads;
    zeroMany<<<nblocks,nthreads, 0, stream>>>(h,nh);
  }

  template<typename Histo, typename T>
  __host__
  void fillOneFromVector(Histo * h, T const * v, uint32_t size, int nthreads, cudaStream_t stream) {
    zero(h,1, nthreads, stream);
    auto nblocks = (size+nthreads-1)/nthreads;
    fillFromVector<<<nblocks,nthreads, 0, stream>>>(h,v,size);
  }

  template<typename Histo, typename T>
  __host__
  void fillManyFromVector(Histo * h, uint32_t nh, T const * v, uint32_t * offsets, uint32_t totSize, int nthreads, cudaStream_t stream) {
    zero(h,nh, nthreads, stream);
    auto nblocks = (totSize+nthreads-1)/nthreads;
    fillFromVector<<<nblocks,nthreads, 0, stream>>>(h,nh,v,offsets);
  }

  template<typename Histo>
  __global__
  void zeroMany(Histo * h, uint32_t nh) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    auto ih = i/Histo::nbins;
    auto k = i - ih*Histo::nbins;
    if (ih<nh) {
      h[ih].nspills=0;
      if(k<Histo::nbins) h[ih].n[k]=0;
    }
  }

  template<typename Histo, typename T>
  __global__
  void fillFromVector(Histo * h,  uint32_t nh, T const * v, uint32_t * offsets) {
     auto i = blockIdx.x*blockDim.x + threadIdx.x;
     if(i>=offsets[nh]) return;
     auto off = upper_bound(offsets,offsets+nh+1,i);
     assert((*off)>0);
     int32_t ih = off-offsets-1;
     assert(ih>=0);
     assert(ih<nh); 
     h[ih].fill(v,i);
  }


  template<typename Histo, typename T>
  __global__
  void fillFromVector(Histo * h, T const * v, uint32_t size) {
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


#endif
