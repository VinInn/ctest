#include <cstdio>
#include<algorithm>
#include<cmath>
#include<cassert>
#include<cstdint>


__global__
void addSimple(uint32_t * p, uint32_t n) {

   auto t = threadIdx.x + blockIdx.x * blockDim.x;
   if (t>=n) return;

   atomicAdd(&p[2+t%4],1);

}

__global__
void addBlock(uint32_t * p, uint32_t n) {

   auto t = threadIdx.x + blockIdx.x * blockDim.x;
   if (t>=n) return;

   __shared__ uint32_t l[8]; 
    if (threadIdx.x==0) for (auto & i:l) i=0;
    __syncthreads();

   atomicAdd(&l[2+t%4],1);
   assert(l[0]==0);
   assert(l[2+t%4]>=1);
   
   __syncthreads();
   if (threadIdx.x==0) for (uint32_t i=0; i<8; ++i) atomicAdd(&p[i],l[i]);

}


__global__
void add(uint32_t * p, uint32_t n) {

   auto t = threadIdx.x + blockIdx.x * blockDim.x;
   if (t>=n) return;

   uint32_t l[8]; for (auto & i:l) i=0;
  
   l[2+t%4]++;
   //atomicAdd(&l[2+t%4],1);
   assert(l[0]==0);
   assert(l[2+t%4]==1);

   for (uint32_t i=0; i<8; ++i) atomicAdd(&p[i],l[i]);

}


__global__
void print(uint32_t * p) {

   for (uint32_t i=0; i<8; ++i) printf("%d ",p[i]);
   printf("\n");

}

#include<iostream>

int main() {

   uint32_t * p=nullptr;
 
   cudaMalloc(&p,8*4);
   assert(p);
   cudaMemset(p,0,8*4);
   print<<<1,1>>>(p);
   cudaDeviceSynchronize();

   addSimple<<<32,128>>>(p,27*127);
   print<<<1,1>>>(p);
   cudaDeviceSynchronize();

   cudaMemset(p,0,8*4);
   addBlock<<<32,128>>>(p,27*127);
   print<<<1,1>>>(p);
   cudaDeviceSynchronize();

   std::cout << "what?" <<std::endl;
   cudaMemset(p,0,8*4);
   add<<<32,128>>>(p,27*127);
   print<<<1,1>>>(p);
   cudaDeviceSynchronize();
   std::cout <<    "what?"    <<std::endl;


   uint32_t h[8] = {0};
   cudaMemcpy(h,p,8*4,cudaMemcpyDeviceToHost);
   cudaDeviceSynchronize();

   std::cout << "host ";
   for (auto i:h) std::cout << i << ' ';
   std::cout << std::endl;

   cudaDeviceSynchronize();

}
