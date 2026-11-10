#pragma once

#include <cstdint>

template <typename T, int N>
__device__ void __forceinline__ prefixScanN(T* uc) {
  auto i = threadIdx.x;
  if constexpr (std::is_same<uint16_t, typename std::remove_cv<T>::type>::value) {
    static_assert(N<=64);
    auto mask = __ballot_sync(0xffffffff, i < (N+1)/2);
    if (i>=(N+1)/2) return;
    uint32_t * c = (uint32_t *)(uc);  // make sure allign is ok
    uint32_t x = c[i];
// first shift by hand
    auto x3 = x>>16;
    x = __vadd2(x,x<<16);
    auto y = __shfl_up_sync(mask, x3, 1);
    if (i >= 1) x= __vadd2(x,y);
#pragma unroll
    for (int offset = 1; offset < (N+1)/2; offset <<= 1) {
      auto y = __shfl_up_sync(mask, x, offset);
      if (i >= offset)
        x= __vadd2(x,y);
    }
    c[i] = x;
  }else {
    static_assert(N<=32);
    auto mask = __ballot_sync(0xffffffff, i < N);
    if (i>=N) return;
    auto x = uc[i];
#pragma unroll
    for (int offset = 1; offset < N; offset <<= 1) {
      auto y = __shfl_up_sync(mask, x, offset);
      if (i >= offset)
        x += y;
    }
    uc[i] = x;
  }
}
