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



template <typename T, int N>
__global__ void testPrefixScanN() {
  __shared__ T c[2*((N+1)/2)];
  auto i = threadIdx.x;
  if (i<N) c[i] = 1;
  if (i==0) printf("testing %d %d %d\n",N,i, c[i]);
  __syncthreads();

  prefixScanN<T,N>(c);
  if (i==N-1) printf("scanned %d %d %d %d\n",N,N-1, c[N-1], c[0]);
  __syncthreads();
  
  if (i==0 && 1 != c[0]) printf(format_traits<T>::failed_msg, N, 0, blockDim.x, c[0],c[0]);
  if (i==0) assert(1 == c[0]);
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
  testPrefixScanN<uint16_t,45><<<1,64>>>();
  testPrefixScanN<uint16_t,64><<<1,64>>>();

  testPrefixScanN<uint32_t,4><<<1,64>>>();
  testPrefixScanN<uint32_t,8><<<1,64>>>();
  testPrefixScanN<uint32_t,9><<<1,64>>>();
  testPrefixScanN<uint32_t,12><<<1,64>>>();
  testPrefixScanN<uint32_t,16><<<1,64>>>();
  testPrefixScanN<uint32_t,17><<<1,64>>>();
  testPrefixScanN<uint32_t,32><<<1,64>>>();
  cudaDeviceSynchronize();

}
