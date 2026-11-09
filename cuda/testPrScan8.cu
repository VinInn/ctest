#include <cstdint>
#include <cassert>
#include <cstdio>

template <typename T>
struct format_traits {
public:
  static const constexpr char *failed_msg = "failed %d %d %d: %d %d\n";
};

template <>
struct format_traits<float> {
public:
  static const constexpr char *failed_msg = "failed %d %d %d: %f %f\n";
};

template <typename T, int N>
__device__ void __forceinline__ prefixScanN(T* c, uint32_t i) {
  static_assert(N<=32);
  auto x = c[i];
  auto laneId = threadIdx.x;
#pragma unroll
  for (int offset = 1; offset < N; offset <<= 1) {
    auto y = __shfl_up_sync(0xffffffff, x, offset);
    if (laneId >= offset)
      x += y;
  }
  c[i] = x;
}


template <typename T, int N>
__global__ void testPrefixScanN() {
  __shared__ T c[N];
  auto i = threadIdx.x;
  c[i] = 1;
  __syncthreads();

  prefixScanN<T,N>(c, i);
  __syncthreads();

  assert(1 == c[0]);
  if (i != 0 && i<N) {
    if (c[i] != c[i - 1] + 1)
      printf(format_traits<T>::failed_msg, N, i, blockDim.x, c[i], c[i - 1]);
    assert(c[i] == c[i - 1] + 1);
    assert(c[i] == i + 1);
  }
}


int main() {

  testPrefixScanN<uint16_t,4><<<1,64>>>();
  testPrefixScanN<uint16_t,8><<<1,64>>>();
  testPrefixScanN<uint16_t,9><<<1,64>>>();
  testPrefixScanN<uint16_t,12><<<1,64>>>();
  testPrefixScanN<uint16_t,16><<<1,64>>>();
  testPrefixScanN<uint16_t,17><<<1,64>>>();
  testPrefixScanN<uint16_t,32><<<1,64>>>();
}
