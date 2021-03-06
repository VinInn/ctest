#include<cstdio>
#include<cassert>
__global__ void doit() {

  auto nt = blockDim.x;
  __shared__ int x[2048];
  x[threadIdx.x]=1;
  x[threadIdx.x+nt]=1;
  __syncthreads();

  int nl=0;
  while (__syncthreads_and(nt)) {
   if(threadIdx.x>=nt) continue;
   ++nl;
   x[threadIdx.x]+=x[threadIdx.x+nt];
   nt = nt/2;
  }

  if (threadIdx.x==0) printf("sum %d in %d for %d\n",x[0],nl,nt);

}


int main() {


  doit<<<1,1024>>>();
  cudaDeviceSynchronize();

}
