#include<hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <cmath>
#include <cstdio>


__device__ int comp(int x) {

  if (0==x) return x;
  return x+comp(x-1);

}


__global__ void sum(int x) {
   auto y = x<0 ? x : comp(x);
   printf("%d %d\n",x,y);
}



int main() {

  int cuda_device = 0;
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, cuda_device);
  printf("HIP Capable: SM %d.%d hardware\n", deviceProp.major, deviceProp.minor);

  hipLaunchKernelGGL(sum, dim3(1), dim3(1), 0, 0, -1);
  hipStreamSynchronize(0);
  hipLaunchKernelGGL(sum, dim3(1), dim3(1), 0, 0, 5);
  hipStreamSynchronize(0);

  return 0;
}
