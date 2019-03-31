#include<cassert>
#include<cstdint>
#include<cmath>
#include<random>
#include<vector>


__global__
void init(int * v,  int * w, int * pc, int n) {

   auto first = blockIdx.x * blockDim.x + threadIdx.x;

   auto & c = *pc; c=0;

   for (int i=first; i<n; i+=gridDim.x*blockDim.x) v[i]=1;

   if (0==blockIdx.x) for (int i=first; i<gridDim.x+1; i+=blockDim.x) w[i]=0;
  
}

__global__
void multiBlockReduction(int * v,  int * w, int * pc, int n) {

   auto first = blockIdx.x * blockDim.x + threadIdx.x;

   auto & c = *pc;  
   
   for (int i=first; i<n; i+=gridDim.x*blockDim.x) atomicAdd(&w[blockIdx.x],v[i]);;

   if (0==threadIdx.x) atomicAdd(pc,1);

   if (0==threadIdx.x) printf("done %d\n",blockIdx.x); 
   if (blockIdx.x>0) return;

   if (0==first) while(c<gridDim.x) {}
   __syncthreads();

   for (int i=first; i<gridDim.x; i+=blockDim.x)
     atomicAdd(&w[gridDim.x],w[i]);

   __syncthreads();

   if (0==first) printf("finished %d\n",w[gridDim.x]);
} 

#include <cuda/api_wrappers.h>
#include<iostream>

int main() {

  int NTOT = 1024*48;

  if (cuda::device::count() == 0) {
    std::cerr << "No CUDA devices on this system" << "\n";
    exit(EXIT_FAILURE);
  }

  auto current_device = cuda::device::current::get();

  auto v_d = cuda::memory::device::make_unique<int[]>(current_device, NTOT);
  auto w_d = cuda::memory::device::make_unique<int[]>(current_device, 1025);
  auto c_d = cuda::memory::device::make_unique<int[]>(current_device, 1);

  init<<<48,256>>>(v_d.get(),w_d.get(),c_d.get(),NTOT);
  multiBlockReduction<<<48,256>>>(v_d.get(),w_d.get(),c_d.get(),NTOT);
  cudaDeviceSynchronize();



  return 0;

}
