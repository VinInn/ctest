#include "CUDATask.h"


__global__ void one(int32_t *d_in, int32_t *d_out,  int32_t n) {

  auto init = [&](int32_t iWork) {
    // standard loop  (iWork instead of blockIdx.x)
    auto first = iWork * blockDim.x + threadIdx.x;
   for (int i=first; i<n; i+=gridDim.x*blockDim.x) { d_in[i]=-1; d_out[i]=-5;}
   d_in[3333]=-4;  // touch it everywhere
   if (15==d_in[1234]) d_in[1234]=33;
   if (15==d_out[200234]) d_out[200234]=33;
  };

  init(blockIdx.x);
}


__global__ void two(int32_t *d_in, int32_t *d_out,  int32_t *one, int32_t *blocks, int32_t n) {

  auto setIt = [&](int32_t iWork) {
    // standard loop  (iWork instead of blockIdx.x)
    auto first = iWork * blockDim.x + threadIdx.x;
   if (0==threadIdx.x) blocks[blockIdx.x]=1;
   for (int i=first; i<n; i+=gridDim.x*blockDim.x) {d_in[i]=5; ++one[i];}
   d_in[5324]=4;  // should fail
   if (15==d_in[10234]) d_in[10234]=33;
   if (15==d_out[10234]) d_out[10234]=33;
  };

  setIt(blockIdx.x);

}

__global__ void three(int32_t *d_in, int32_t *d_out,  int32_t n) {

  auto testIt1 = [&](int32_t iWork) {
    // standard loop  (iWork instead of blockIdx.x)
    auto first = (gridDim.x-iWork-1)  * blockDim.x + threadIdx.x;
    for (int i=first; i<n; i+=gridDim.x*blockDim.x) if (5==d_in[i]) d_out[i]=5;
  };

  testIt1(blockIdx.x);

}

template<int N>
__global__ void testTask(int32_t *d_in, int32_t *d_out,  int32_t *one, int32_t * blocks, int32_t n, CUDATask * task) {

  auto voidTail = [](){};


  auto init = [&](int32_t iWork) {
    // standard loop  (iWork instead of blockIdx.x)
    auto first = iWork * blockDim.x + threadIdx.x;
   for (int i=first; i<n; i+=gridDim.x*blockDim.x) { d_in[i]=-1; d_out[i]=-5;}  
   d_in[3333]=-4;  // touch it everywhere
   if (15==d_in[1234]) d_in[1234]=33;
   if (15==d_out[200234]) d_out[200234]=33;

  };


  auto setIt = [&](int32_t iWork) {
    // standard loop  (iWork instead of blockIdx.x)
    auto first = iWork * blockDim.x + threadIdx.x;
   if (0==threadIdx.x) blocks[blockIdx.x]=1;
   for (int i=first; i<n; i+=gridDim.x*blockDim.x) {d_in[i]=5; ++one[i];}
   d_in[5324]=4;  // should fail
   if (15==d_in[10234]) d_in[10234]=33;
   if (15==d_out[10234]) d_out[10234]=33;

  };

  auto testIt1 = [&](int32_t iWork) {
    // standard loop  (iWork instead of blockIdx.x)
    auto first = (gridDim.x-iWork-1)  * blockDim.x + threadIdx.x;
    for (int i=first; i<n; i+=gridDim.x*blockDim.x) if (5==d_in[i]) d_out[i]=5;
  };

  task[0].doit(init,voidTail);
  task[1].doit(setIt,voidTail);
  task[2].doit(testIt1,voidTail);

}


__global__ void verify(int32_t *d_out, int32_t * one, int32_t n) {
   auto first = blockIdx.x * blockDim.x + threadIdx.x;
   for (int i=first; i<n; i+=gridDim.x*blockDim.x) {
     if (5!=d_out[i]) printf("out failed %d %d/%d\n",i,blockIdx.x,threadIdx.x);
     if (1!=one[i]) printf("one failed %d %d/%d\n",i,blockIdx.x,threadIdx.x);
   }
}


#include <iostream>

#include "cudaCheck.h"
#include "requireDevices.h"
#include <chrono>

using namespace std::chrono;


int main() {

  cms::cudatest::requireDevices();

  int32_t *d_in;
  int32_t *d_out1;
  int32_t *d_out2;

  int32_t num_items = 1000*1000;

  cudaCheck(cudaMalloc(&d_in, num_items * sizeof(uint32_t)));
  cudaCheck(cudaMalloc(&d_out1, num_items * sizeof(uint32_t)));
  cudaCheck(cudaMalloc(&d_out2, num_items * sizeof(uint32_t)));

  auto nthreads = 256;
  auto nblocks = (num_items + nthreads - 1) / nthreads;

  int32_t * blocks;
  int32_t * h_blocks;
  cudaCheck(cudaMalloc(&blocks, nblocks * sizeof(uint32_t)));
  cudaCheck(cudaMemset(blocks, 0, nblocks * sizeof(uint32_t)));
  cudaCheck(cudaMallocHost(&h_blocks, nblocks * sizeof(uint32_t)));

  CUDATask * task;  // 3 of them
  cudaCheck(cudaMalloc(&task, 3*sizeof(CUDATask)));
  cudaCheck(cudaMemset(task, 0, 3*sizeof(CUDATask)));

  cudaCheck(cudaMemset(d_in, 0, num_items*sizeof(int32_t)));
  cudaCheck(cudaMemset(d_out1, 0, num_items*sizeof(int32_t)));
  cudaCheck(cudaMemset(d_out2, 0, num_items*sizeof(int32_t)));

  {
  std::cout << "scheduling " << nblocks << " blocks of " << nthreads << " threads"<< std::endl;
  cudaDeviceSynchronize();
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  one<<<nblocks, nthreads, 0>>>(d_in, d_out1, num_items);
  two<<<nblocks, nthreads, 0>>>(d_in, d_out1, d_out2, blocks, num_items);
  three<<<nblocks, nthreads, 0>>>(d_in, d_out1, num_items);
  cudaDeviceSynchronize();
  high_resolution_clock::time_point t2 = high_resolution_clock::now();

  cudaCheck(cudaGetLastError());
  verify<<<nblocks, nthreads, 0>>>(d_out1, d_out2, num_items);
  cudaCheck(cudaGetLastError());
  cudaDeviceSynchronize();
  cudaMemcpy(h_blocks, blocks, nblocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  int s=0; for (int i=0; i<nblocks;++i) s += h_blocks[i];
  std::cout << "standard kernel used " << s    << " blocks" << std::endl;
  auto delta = duration_cast<duration<double>>(t2 - t1).count();
  std::cout << "three kernels took " << delta << std::endl;
  }


  {
  cudaCheck(cudaMemset(d_in, 0, num_items*sizeof(int32_t)));
  cudaCheck(cudaMemset(d_out1, 0, num_items*sizeof(int32_t)));
  cudaCheck(cudaMemset(d_out2, 0, num_items*sizeof(int32_t)));
  cudaCheck(cudaMemset(task, 0, 3*sizeof(CUDATask)));
  cudaCheck(cudaMemset(blocks, 0, nblocks * sizeof(uint32_t)));

  std::cout << "scheduling " << nblocks << " blocks of " << nthreads << " threads"<< std::endl;
  cudaDeviceSynchronize();
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  testTask<1> <<<nblocks, nthreads, 0>>>(d_in, d_out1, d_out2, blocks, num_items, task);
  cudaDeviceSynchronize();
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  cudaCheck(cudaGetLastError());
  verify<<<nblocks, nthreads, 0>>>(d_out1, d_out2, num_items);
  cudaCheck(cudaGetLastError());
  cudaDeviceSynchronize(); 
  cudaMemcpy(h_blocks, blocks, nblocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  int s=0; for (int i=0; i<nblocks;++i) s += h_blocks[i];
  std::cout << "task kernel used " << s    << " blocks" << std::endl;
  auto delta = duration_cast<duration<double>>(t2 - t1).count();
  std::cout << "task kernel took " << delta << std::endl;
  }

  {
     nblocks /= 32;
  cudaCheck(cudaMemset(d_in, 0, num_items*sizeof(int32_t)));
  cudaCheck(cudaMemset(d_out1, 0, num_items*sizeof(int32_t)));
  cudaCheck(cudaMemset(d_out2, 0, num_items*sizeof(int32_t)));
  cudaCheck(cudaMemset(task, 0, 3*sizeof(CUDATask)));
  cudaCheck(cudaMemset(blocks, 0, nblocks * sizeof(uint32_t)));

  std::cout << "scheduling " << nblocks << " blocks of " << nthreads << " threads"<< std::endl;
  cudaDeviceSynchronize();
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  testTask<2> <<<nblocks, nthreads, 0>>>(d_in, d_out1, d_out2, blocks, num_items, task);
  cudaDeviceSynchronize();
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  cudaCheck(cudaGetLastError());
  verify<<<nblocks, nthreads, 0>>>(d_out1, d_out2, num_items);
  cudaCheck(cudaGetLastError());
  cudaDeviceSynchronize();
  cudaMemcpy(h_blocks, blocks, nblocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  int s=0; for (int i=0; i<nblocks;++i) s += h_blocks[i];
  std::cout << "task kernel used " << s << " blocks" << std::endl;
  auto delta = duration_cast<duration<double>>(t2 - t1).count();
  std::cout << "task kernel took " << delta << std::endl;
  }

  return 0;
};
 
