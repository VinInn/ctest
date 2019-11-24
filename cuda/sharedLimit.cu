__device__
void foo1(double * x) {
  __shared__ double local[4086];

  local[blockIdx.x] = x[threadIdx.x];
  __syncthreads();

  x[threadIdx.x] =  local[threadIdx.x];

}

__device__
void foo2(double * x) {
  __shared__ double local[4086];

  local[blockIdx.x] = x[threadIdx.x];
  __syncthreads();

  x[threadIdx.x] =  local[threadIdx.x];

}

__global__
void go(double * x) {

  foo1(x);
  __syncthreads();
  foo2(x);
}



#include<cuda.h>

int main() {

  double * x;
  cudaMalloc(&x,4096*4);

  go<<<1,1024>>>(x);

  cudaDeviceSynchronize();

  return 0;
}


