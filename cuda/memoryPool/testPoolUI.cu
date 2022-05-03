#include "cudaMemoryPool.h"

#include<atomic>
#include<thread>
#include<mutex>
#include<chrono>

int main() {


  auto start = std::chrono::high_resolution_clock::now();

  const int NUMTHREADS=24;

  printf ("Using CUDA %d\n",CUDART_VERSION);
  int cuda_device = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, cuda_device);
  printf("CUDA Capable: SM %d.%d hardware\n", deviceProp.major, deviceProp.minor);


  cudaStream_t streams[NUMTHREADS];


  for (int i = 0; i < NUMTHREADS; i++) {
     cudaStreamCreate(&(streams[i]));
  }

  memoryPool::cuda::dumpStat();

  auto & stream = streams[0]; 

  {
    auto p = memoryPool::cuda::device::make_unique<int>(20,stream);

    memoryPool::cuda::dumpStat();
  }


  {
     memoryPool::Deleter deleter(std::make_shared<memoryPool::cuda::BundleDelete>(stream,memoryPool::cuda::getPool(true)));

     auto p0 = memoryPool::cuda::device::make_unique<int>(20,deleter);
     auto p1 = memoryPool::cuda::device::make_unique<double>(20,deleter);
     auto p2 = memoryPool::cuda::device::make_unique<bool>(20,deleter);
     auto p3 = memoryPool::cuda::device::make_unique<int>(20,deleter);

     memoryPool::cuda::dumpStat();
  }

  cudaStreamSynchronize(stream);
  memoryPool::cuda::dumpStat();

  return 0;
}
