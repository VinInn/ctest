
#include "prefixScanN.h"
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
