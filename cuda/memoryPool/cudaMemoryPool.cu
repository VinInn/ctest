#include "cudaMemoryPool.h"

#include "FastPoolAllocator.h"

FastPoolAllocator<CudaDeviceAlloc,1024*1024>  devicePool;
FastPoolAllocator<CudaHostAlloc,1024*1024>  hostPool;


namespace memoryPool {
  namespace cuda {

    void dumpStat() {
       std::cout << "device pool" << std::endl;
       devicePool.dumpStat();
       std::cout << "host pool" << std::endl;
       hostPool.dumpStat();

    }

    struct Payload {
      std::vector<int> buckets; bool onDevice;
    };

    // generic callback
    void CUDART_CB freeCallback(void * p){
      auto payload = (Payload*)(p);
      auto onDevice = payload->onDevice;
      auto const & buckets = payload->buckets;
        std::cout << "do free " << buckets.size() << ' ' << buckets[0] << " on " << onDevice << std::endl;
        for (auto i :  buckets) {
          if (onDevice) devicePool.free(i);
          else hostPool.free(i);
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
      std::cout << "schedule free " << buckets.size() << ' ' << buckets[0] << std::endl;
      auto payload = new Payload{std::move(buckets), onDevice};
      cudaLaunchHostFunc (stream, freeCallback, payload);
    }

  }
}
