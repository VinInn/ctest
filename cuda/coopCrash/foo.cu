#include <cstdio>

#include <cooperative_groups.h>
using namespace cooperative_groups;


__global__ 
void foo() {
  grid_group grid = this_grid();
  if (0==blockIdx.x) printf("Hello\n");
  grid.sync();
  if (1==blockIdx.x) printf("Hello again\n");
  grid.sync();
}

#include<cuda.h>
#include "cudaCheck.h"
#include "launch.h"

void fooWrapper() {

  cms::cuda::launch_cooperative(foo,{1,1});
  cudaCheck(cudaGetLastError());
  cudaDeviceSynchronize();
 

}


void docheck() {
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
}
