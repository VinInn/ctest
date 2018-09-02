#include <cstdio>
#include <cmath>
#include <algorithm>

__global__ void warpReduce() {
    int laneId = threadIdx.x & 0x1f;
    // Seed starting value as inverse lane ID
    int value = 31 - laneId;

    // Use XOR mode to perform butterfly reduction
    for (int i=16; i>=1; i/=2)
        value += __shfl_xor_sync(0xffffffff, value, i, 32);

    // "value" now contains the sum across all threads
    printf("Thread %d final value = %d\n", threadIdx.x, value);
}

__global__ void warpMin() {
  
    __shared__ int x[256];
    int laneId = threadIdx.x & 0x1f;
    int value = 15 - threadIdx.x;
    x[threadIdx.x] = value;

    if (threadIdx.x < 60) {
    // auto mask = __activemask(); // not needed!
    // Use XOR mode to perform butterfly reduction
      for (int i=16; i>=1; i/=2)
          value = std::min(value,__shfl_xor_sync(0xffffffff, value, i, 32));
    }

    // "value" now contains the sum across all threads
    printf("Thread %d final value = %d\n", threadIdx.x, value);

   __shared__ int min;
   __shared__ int minloc;
   min = value;
   __syncthreads();
   if (laneId==0) atomicMin(&min,value);
   if (x[threadIdx.x]==min) minloc=threadIdx.x;
   __syncthreads();

   if (threadIdx.x==0) printf("final value = %d @ %d \n", min,minloc);

}


int main() {
    warpReduce<<< 1, 32 >>>();
    warpMin<<< 1, 64 >>>();
    cudaDeviceSynchronize();

    return 0;
}
