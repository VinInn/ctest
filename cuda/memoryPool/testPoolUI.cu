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
    auto pd = memoryPool::cuda::make_unique<int>(20,stream,memoryPool::onDevice);
    auto ph = memoryPool::cuda::make_unique<int>(20,stream,memoryPool::onHost);

    memoryPool::cuda::dumpStat();
  }


  {
     memoryPool::Deleter devDeleter(std::make_shared<memoryPool::cuda::BundleDelete>(stream,memoryPool::onDevice));
     memoryPool::Deleter hosDeleter(std::make_shared<memoryPool::cuda::BundleDelete>(stream,memoryPool::onHost));

     auto p0 = memoryPool::cuda::make_unique<int>(20,devDeleter);
     auto p1 = memoryPool::cuda::make_unique<double>(20,devDeleter);
     auto p2 = memoryPool::cuda::make_unique<bool>(20,devDeleter);
     auto p3 = memoryPool::cuda::make_unique<int>(20,devDeleter);

     auto hp0 = memoryPool::cuda::make_unique<int>(20,hosDeleter);
     auto hp1 = memoryPool::cuda::make_unique<double>(20,hosDeleter);
     auto hp2 = memoryPool::cuda::make_unique<bool>(20,hosDeleter);
     auto hp3 = memoryPool::cuda::make_unique<int>(20,hosDeleter);

     memoryPool::cuda::dumpStat();
  }

  cudaStreamSynchronize(stream);
  memoryPool::cuda::dumpStat();

  return 0;
}
