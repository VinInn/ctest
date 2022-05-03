#include "cudaMemoryPool.h"

#include "FastPoolAllocator.h"

FastPoolAllocatorImpl<CudaDeviceAlloc>  devicePool(1024);
FastPoolAllocatorImpl<CudaHostAlloc>  hostPool(1024);


namespace memoryPool {
  namespace cuda {

    void dumpStat() {
       std::cout << "device pool" << std::endl;
       devicePool.dumpStat();
       std::cout << "host pool" << std::endl;
       hostPool.dumpStat();

    }

    struct Payload {
      FastPoolAllocator * pool;
      std::vector<int> buckets;
    };

    // generic callback
    void CUDART_CB freeCallback(void * p){
      auto payload = (Payload*)(p);
      auto & pool = *(payload->pool);
      auto const & buckets = payload->buckets;
        std::cout << "do free " << buckets.size();
        if (!buckets.empty()) std::cout  << ' ' << buckets.front() << ' ' << buckets.back();
        std::cout << std::endl;
        for (auto i :  buckets) {
          pool.free(i);
        }
      delete payload;
    }

    // allocate either on current device or on host
    std::pair<void *,int> alloc(uint64_t size, bool onDevice) {
       int i;
       void * p;
       if (onDevice) {
         i = devicePool.alloc(size);
         p = devicePool.pointer(i);
       } else {
         i = hostPool.alloc(size);
         p = hostPool.pointer(i);
       }
       return std::pair<void *,int>(p,i);
    }

    // schedule free
    void free(cudaStream_t stream, std::vector<int> buckets, bool onDevice) {
      // free
      std::cout << "schedule free " << buckets.size() << ' ';
      if (!buckets.empty()) std::cout << buckets[0]; 
      std::cout << std::endl;
      auto payload = new Payload{onDevice ? (FastPoolAllocator *)(&devicePool) : (FastPoolAllocator *)(&hostPool), std::move(buckets)};
      cudaLaunchHostFunc (stream, freeCallback, payload);
    }

  }
}
