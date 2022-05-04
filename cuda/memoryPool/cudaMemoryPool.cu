#include "cudaMemoryPool.h"

#include "SimplePoolAllocator.h"

SimplePoolAllocatorImpl<CudaDeviceAlloc>  devicePool(1024);
SimplePoolAllocatorImpl<CudaHostAlloc>  hostPool(1024);


namespace memoryPool {
  namespace cuda {

    void dumpStat() {
       std::cout << "device pool" << std::endl;
       devicePool.dumpStat();
       std::cout << "host pool" << std::endl;
       hostPool.dumpStat();

    }


   SimplePoolAllocator * getPool(Where where) {
      return onDevice==where ?  (SimplePoolAllocator *)(&devicePool) : (SimplePoolAllocator *)(&hostPool);
   }

    struct Payload {
      SimplePoolAllocator * pool;
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
    std::pair<void *,int> alloc(uint64_t size, SimplePoolAllocator & pool) {
       int i = pool.alloc(size);
       void * p = pool.pointer(i);
       return std::pair<void *,int>(p,i);
    }

    // schedule free
    void free(cudaStream_t stream, std::vector<int> buckets, SimplePoolAllocator & pool) {
      // free
      std::cout << "schedule free " << buckets.size() << ' ';
      if (!buckets.empty()) std::cout << buckets[0]; 
      std::cout << std::endl;
      auto payload = new Payload{&pool, std::move(buckets)};
      cudaLaunchHostFunc (stream, freeCallback, payload);
    }

  }
}
