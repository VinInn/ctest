#include <cstdio>
#include<algorithm>
#include<cmath>

__global__ void bar(int n) {
    int laneId = threadIdx.x & 0x1f;
    __shared__ int res1;
    __shared__ int res2;
    __shared__ int res3;
    __shared__ int v[1024];
    res1= res2 = res3=0;
    __syncthreads();

    for (auto i=threadIdx.x; i<n; i+=blockDim.x) {
      if (laneId%3==1) atomicMax(&res1,i%laneId);
      if (laneId%3==2) atomicAdd(&res2,i%laneId);
      if (laneId%3==2) {
        auto old = atomicAdd(&res3,1);
        v[old] = i%laneId;
      }  
    }

    __syncthreads();

    if (threadIdx.x==0) printf("res %d %d %d\n", res1,res2,res3); 
    if (threadIdx.x==0) printf("v %d %d %d\n", v[0],v[res3/2],v[res3-1]);
}



int main() {

  bar<<<64,128,0>>>(154);
  cudaDeviceSynchronize();
}
