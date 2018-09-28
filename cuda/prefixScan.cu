#include <cstdint>
#include <cassert>

template<typename T>
__device__
void 
__forceinline__
warpPrefixScan(T * c, uint32_t i) {
   auto x = c[i];
   auto laneId = threadIdx.x & 0x1f;
   #pragma unroll
   for( int offset = 1 ; offset < 32 ; offset <<= 1 ) {
     auto y = __shfl_up_sync(0xffffffff,x, offset);
     if(laneId >= offset) x += y;
   }
   c[i] = x;
}

// limited to 32*32 elements....
template<typename T>
__device__
void
__forceinline__
blockPrefixScan(T * c, uint32_t size, T* ws) {
  assert(size<=1024);
  assert(0==blockDim.x%32);

  auto first = threadIdx.x;

  for (auto i=first; i<size; i+=blockDim.x) {
    warpPrefixScan(c,i);
    auto laneId = threadIdx.x & 0x1f;
    auto warpId = i/32;
    assert(warpId<32);
    if (31==laneId) ws[warpId]=c[i];
  }
  __syncthreads();
  if (size<=32) return;
  if (threadIdx.x<32) warpPrefixScan(ws,threadIdx.x);
  __syncthreads();
  for (auto i=first+32; i<size; i+=blockDim.x) {
    auto warpId = i/32;
    c[i]+=ws[warpId-1];
  }
  __syncthreads();
}

template<typename T>
__global__
void testPrefixScan(uint32_t size) {

  __shared__ T ws[32];
  __shared__ T c[1024];
  auto first = threadIdx.x;
  for (auto i=first; i<size; i+=blockDim.x) c[i]=1;
  __syncthreads();

  blockPrefixScan(c, size, ws);

  assert(1==c[0]);
  for (auto i=first+1; i<size; i+=blockDim.x) {
    if (c[i]!=c[i-1]+1) printf("failed %d %d %d: %d %d\n",size, i, blockDim.x, c[i],c[i-1]);
    assert(c[i]==c[i-1]+1); assert(c[i]==i+1);
  }
}


template<typename T>
__global__
void testWarpPrefixScan(uint32_t size) {
  assert(size<=32);
  __shared__ T c[1024];
  auto i = threadIdx.x;
  c[i]=1;
  __syncthreads();

  warpPrefixScan(c,i);
 __syncthreads();

  assert(1==c[0]);
  if(i!=0) {
    if (c[i]!=c[i-1]+1) printf("failed %d %d %d: %d %d\n",size, i, blockDim.x, c[i],c[i-1]);
    assert(c[i]==c[i-1]+1); assert(c[i]==i+1);
  }
}


#include<iostream>
int main() {

  std::cout << "warp 32" << std::endl;
  testWarpPrefixScan<int><<<1,32>>>(32);
  cudaDeviceSynchronize();
  std::cout << "warp 16" << std::endl;
  testWarpPrefixScan<int><<<1,32>>>(16);
  cudaDeviceSynchronize();
  std::cout << "warp 5" << std::endl;
  testWarpPrefixScan<int><<<1,32>>>(5);
  cudaDeviceSynchronize();

  for(int bs=32; bs<=1024; bs+=32) {
  std::cout << "bs " << bs << std::endl;
  for (int j=1;j<=1024; ++j) {
   std::cout << j << std::endl;
   testPrefixScan<uint16_t><<<1,bs>>>(j);
  cudaDeviceSynchronize();
//   testPrefixScan<float><<<1,bs>>>(j);
//  cudaDeviceSynchronize();
  }}
  cudaDeviceSynchronize();

  return 0;
}
