#include "cudaMemoryPool.h"



namespace memoryPool {
  namespace cuda {

    // allocate either on current device or on host
    std::pair<void *,int> alloc(uint64_t size, bool onDevice) {
       return std::pair<void *,int>(nullptr,-1);
    }

    // schedule free
    void free(cudaStream_t stream, int * bucket, int n, bool onDevice) {

    }

  }
}
