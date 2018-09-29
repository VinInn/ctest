#include <cstdint>
#include <cassert>


__global__
void testPrefixScan(uint32_t size) {

  __shared__ uint16_t c[1024];
  auto first = threadIdx.x;
  for (auto i=first; i<size; i+=blockDim.x) c[i]=1;
  __syncthreads();

  unsigned mask = __ballot_sync(0xffffffff,first<size);
//   unsigned mask = 0xffffffff;
#ifdef NOLOOP
  auto i=first;
#else
  for (auto i=first; i<size; i+=blockDim.x)
#endif
  {
    auto x = c[i];
    auto laneId = threadIdx.x & 0x1f;
    #pragma unroll
    for( int offset = 1 ; offset < 32 ; offset <<= 1 ) {
      auto y = __shfl_up_sync(mask,x, offset);
      if(laneId >= offset) x += y;
    }
   c[i] = x;
   mask = __ballot_sync(mask,i+blockDim.x<size);
  }
  __syncthreads();
}

#include<iostream>
int main() {
  testPrefixScan<<<1,32>>>(45);
  cudaDeviceSynchronize();

  return 0;
}
