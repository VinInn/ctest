#include<cstdint>
#include<cassert>
#include<cstdio>

template<int DIM>
__global__
void bha() {
__shared__ uint16_t a[DIM];
}



__device__  
void radixSort(int16_t * v, uint16_t * index, uint32_t * offsets) {
    
  constexpr int d = 8, w = 16;
  constexpr int sb = 1<<d;

  constexpr int MaxSize = 256*32;
  __shared__ uint16_t ind2[MaxSize];
  __shared__ int32_t c[sb], ct[sb], cu[sb];
  __shared__ uint32_t firstNeg;    
  __shared__ bool go; 

  // later add offset
  auto a = v+offsets[blockIdx.x];   
  auto ind = index+offsets[blockIdx.x];;
  auto size = offsets[blockIdx.x+1]-offsets[blockIdx.x];
  
  assert(size<=MaxSize); 
  assert(blockDim.x==sb);  
}
