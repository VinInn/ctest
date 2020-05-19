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
   
   for (int i=first; i<n; i+=gridDim.x*blockDim.x) atomicAdd_block(&w[blockIdx.x],v[i]);;

   if (0==threadIdx.x) atomicAdd(pc,1);

   if (0==threadIdx.x) printf("done %d\n",blockIdx.x); 
   if (blockIdx.x>0) return;

   if (0==first) while(c<gridDim.x) {}
   __syncthreads();

   for (int i=first; i<gridDim.x; i+=blockDim.x)
     atomicAdd_grid(&w[gridDim.x],w[i]);

   __syncthreads();

   if (0==first) printf("finished %d\n",w[gridDim.x]);
} 

#include<iostream>

int main() {


  int NTOT = 1024*48;

  std::cout << "summing " << NTOT << " ones" << std::endl;

  int *  v_d;  cudaMalloc(&v_d, sizeof(int)*NTOT);
  int *  w_d;  cudaMalloc(&w_d, sizeof(int)*1025);
  int *  c_d;  cudaMalloc(&c_d, sizeof(int));

  init<<<48,256>>>(v_d,w_d,c_d,NTOT);
  multiBlockReduction<<<48,256>>>(v_d,w_d,c_d,NTOT);
  cudaDeviceSynchronize();



  return 0;

}
