#ifndef HeterogeneousCore_CUDAUtilities_interface_OneToManyAssoc_h
#define HeterogeneousCore_CUDAUtilities_interface_OneToManyAssoc_h

#include <algorithm>
#ifndef __CUDA_ARCH__
#include <atomic>
#endif  // __CUDA_ARCH__
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "HeterogeneousCore/CUDAUtilities/interface/AtomicPairCounter.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudastdAlgorithm.h"
#include "HeterogeneousCore/CUDAUtilities/interface/prefixScan.h"
#include "HeterogeneousCore/CUDAUtilities/interface/FlexiStorage.h"

namespace cms {
  namespace cuda {

    template <typename Assoc>
    __global__ void zeroAndInit(
        Assoc *h, typename Assoc::index_type *mem, int s, typename Assoc::Counter *c, int32_t n) {
      int first = blockDim.x * blockIdx.x + threadIdx.x;

      if (0 == first) {
        h->psws = 0;
        h->initStorage(c, n, mem, s);
      }
      __syncthreads();
      for (int i = first, nt = h->totOnes(); i < nt; i += gridDim.x * blockDim.x) {
        h->off[i] = 0;
      }
    }

    template <typename Assoc>
    inline __attribute__((always_inline)) void launchZero(Assoc *__restrict__ h,
                                                          typename Assoc::index_type *mem,
                                                          int s,
                                                          typename Assoc::Counter *c,
                                                          int32_t n,
                                                          cudaStream_t stream
#ifndef __CUDACC__
                                                          = cudaStreamDefault
#endif
    ) {
      if constexpr (Assoc::ctCapacity() < 0) {
        assert(mem);
        assert(s > 0);
      }
      auto nOnes = Assoc::ctNOnes();
      if constexpr (Assoc::ctNOnes() < 0) {
        assert(c);
        assert(n > 0);
        nOnes = n;
      }
      assert(nOnes > 0);
#ifdef __CUDACC__
      auto nthreads = 1024;
      auto nblocks = (nOnes + nthreads - 1) / nthreads;
      zeroAndInit<<<nblocks, nthreads, 0, stream>>>(h, mem, s, c, n);
      cudaCheck(cudaGetLastError());
#else
      h->initStorage(c, n, mem, s);
      h->zero();
      h->psws = 0;
#endif
    }

    template <typename Assoc>
    inline __attribute__((always_inline)) void launchFinalize(Assoc *__restrict__ h,
                                                              typename Assoc::Counter *c,
                                                              int32_t n,
                                                              cudaStream_t stream
#ifndef __CUDACC__
                                                              = cudaStreamDefault
#endif
    ) {
#ifdef __CUDACC__
      using Counter = typename Assoc::Counter;
      auto nOnes = Assoc::ctNOnes();
      Counter *poff = (Counter *)((char *)(h) + offsetof(Assoc, off));
      if constexpr (Assoc::ctNOnes() < 0) {
        assert(c);
        assert(n > 0);
        nOnes = n;
        poff = c;
      }
      assert(nOnes > 0);
      int32_t *ppsws = (int32_t *)((char *)(h) + offsetof(Assoc, psws));
      auto nthreads = 1024;
      auto nblocks = (nOnes + nthreads - 1) / nthreads;
      multiBlockPrefixScan<<<nblocks, nthreads, sizeof(int32_t) * nblocks, stream>>>(poff, poff, nOnes, ppsws);
      cudaCheck(cudaGetLastError());
#else
      h->finalize();
#endif
    }

    template <typename Assoc>
    __global__ void finalizeBulk(AtomicPairCounter const *apc, Assoc *__restrict__ assoc) {
      assoc->bulkFinalizeFill(*apc);
    }

    template <typename I,    // type stored in the container (usually an index in a vector of the input values)
              int32_t ONES,  // number of "Ones" If -1 is initialized at runtime using external storage
              int32_t SIZE   // max number of element. If -1 is initialized at runtime using external storage
              >
    class OneToManyAssoc {
    public:
      using Counter = uint32_t;

      using CountersOnly = OneToManyAssoc<I, ONES, 0>;

      using index_type = I;

      static constexpr uint32_t ilog2(uint32_t v) {
        constexpr uint32_t b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
        constexpr uint32_t s[] = {1, 2, 4, 8, 16};

        uint32_t r = 0;  // result of log2(v) will go here
        for (auto i = 4; i >= 0; i--)
          if (v & b[i]) {
            v >>= s[i];
            r |= s[i];
          }
        return r;
      }

      static constexpr int32_t ctNOnes() { return ONES; }
      constexpr auto totOnes() const { return off.capacity(); }
      constexpr auto nOnes() const { return totOnes() - 1; }
      static constexpr int32_t ctCapacity() { return SIZE; }
      constexpr auto capacity() const { return content.capacity(); }

      __host__ __device__ void initStorage(Counter *c, int32_t n, I *d, int32_t s) {
        if constexpr (ctNOnes() < 0) {
          assert(c);
          assert(n > 0);
          off.init(c, n);
        }
        if constexpr (ctCapacity() < 0) {
          assert(d);
          assert(s > 0);
          content.init(d, s);
        }
      }

      __host__ __device__ void initStorage(I *d, int32_t s) { content.init(d, s); }

      __host__ __device__ void zero() {
        for (int32_t i = 0; i < totOnes(); ++i) {
          off[i] = 0;
        }
      }

      __host__ __device__ __forceinline__ void add(CountersOnly const &co) {
        for (int32_t i = 0; i < totOnes(); ++i) {
#ifdef __CUDA_ARCH__
          atomicAdd(off.data() + i, co.off[i]);
#else
          auto &a = (std::atomic<Counter> &)(off[i]);
          a += co.off[i];
#endif
        }
      }

      static __host__ __device__ __forceinline__ uint32_t atomicIncrement(Counter &x) {
#ifdef __CUDA_ARCH__
        return atomicAdd(&x, 1);
#else
        auto &a = (std::atomic<Counter> &)(x);
        return a++;
#endif
      }

      static __host__ __device__ __forceinline__ uint32_t atomicDecrement(Counter &x) {
#ifdef __CUDA_ARCH__
        return atomicSub(&x, 1);
#else
        auto &a = (std::atomic<Counter> &)(x);
        return a--;
#endif
      }

      __host__ __device__ __forceinline__ void count(int32_t b) {
        assert(b < nOnes());
        atomicIncrement(off[b]);
      }

      __host__ __device__ __forceinline__ void fill(int32_t b, index_type j) {
        assert(b < nOnes());
        auto w = atomicDecrement(off[b]);
        assert(w > 0);
        content[w - 1] = j;
      }

      __host__ __device__ __forceinline__ int32_t bulkFill(AtomicPairCounter &apc, index_type const *v, uint32_t n) {
        auto c = apc.add(n);
        if (int(c.m) >= nOnes())
          return -int32_t(c.m);
        off[c.m] = c.n;
        for (uint32_t j = 0; j < n; ++j)
          content[c.n + j] = v[j];
        return c.m;
      }

      __host__ __device__ __forceinline__ void bulkFinalize(AtomicPairCounter const &apc) {
        off[apc.get().m] = apc.get().n;
      }

      __host__ __device__ __forceinline__ void bulkFinalizeFill(AtomicPairCounter const &apc) {
        int m = apc.get().m;
        auto n = apc.get().n;
        if (m >= nOnes()) {  // overflow!
          off[nOnes()] = uint32_t(off[nOnes() - 1]);
          return;
        }
        auto first = m + blockDim.x * blockIdx.x + threadIdx.x;
        for (int i = first; i < totOnes(); i += gridDim.x * blockDim.x) {
          off[i] = n;
        }
      }

      __host__ __device__ __forceinline__ void finalize(Counter *ws = nullptr) {
        assert(off[totOnes() - 1] == 0);
        blockPrefixScan(off.data(), totOnes(), ws);
        assert(off[totOnes() - 1] == off[totOnes() - 2]);
      }

      constexpr auto size() const { return uint32_t(off[totOnes() - 1]); }
      constexpr auto size(uint32_t b) const { return off[b + 1] - off[b]; }

      constexpr index_type const *begin() const { return content.data(); }
      constexpr index_type const *end() const { return begin() + size(); }

      constexpr index_type const *begin(uint32_t b) const { return content.data() + off[b]; }
      constexpr index_type const *end(uint32_t b) const { return content.data() + off[b + 1]; }

      FlexiStorage<Counter, ONES> off;
      int32_t psws;  // prefix-scan working space
      FlexiStorage<index_type, SIZE> content;
    };

  }  // namespace cuda
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h
