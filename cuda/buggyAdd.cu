#include <cstdio>
#include<algorithm>
#include<cmath>
#include<cassert>
#include<cstdint>

__global__
void add(uint32_t * p, uint32_t n) {

   auto t = threadIdx.x + blockIdx.x * blockDim.x;
   if (t>=n) return;

   uint32_t l[8]; for (auto & i:l) i=0;
  
   atomicAdd(l+(2+t%4),1);

   for (uint32_t i=0; i<8; ++i) atomicAdd(&p[i],l[i]);

}

#include<iostream>

int main() {

   uint32_t * p;
 
   cudaMalloc(&p,8*4);
   cudaMemset(p,0,8*4);
   add<<<32,128>>>(p,27*127);

   uint32_t h[8];
   cudaMemcpy(h,p,8*4,cudaMemcpyDeviceToHost);

   for (auto i:h) std::cout << i << ' ';
   std::cout << std::endl;

   cudaDeviceSynchronize();

}
