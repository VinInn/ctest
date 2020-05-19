#ifndef HeterogeneousCore_CUDAUtilities_interface_CUDATask_h
#define HeterogeneousCore_CUDAUtilities_interface_CUDATask_h

#include <cstdint>

#include "cudaCompat.h"
#include "cuda_assert.h"

class CUDATask {

public: 

  // better to be called in the tail of the previous task...
  __device__ void __forceinline__ zero() { nWork=0; nDone=0; allDone=0;}

  template<typename BODY, typename TAIL>
  __device__ void __forceinline__ doit(BODY body, TAIL tail) {

      __shared__ int iWork;
      bool done=false;
      __shared__ bool isLastBlockDone;

     isLastBlockDone = false;

      while(__syncthreads_and(!done)) {
        if (0 == threadIdx.x) {
          iWork = atomicAdd(&nWork, 1) ;
        }
        __syncthreads();

        assert(iWork >=0);

        done = iWork >=int(gridDim.x);

        if (!done) {
          body(iWork);

          // count blocks that finished
          if (0 == threadIdx.x) {
            auto value = atomicAdd(&nDone, 1);  // block counter
            isLastBlockDone = (value == (int(gridDim.x) - 1));
          }

        } // done
      }  // while

      if (isLastBlockDone) {
        assert(0==(allDone));

        assert(int(gridDim.x) == nDone);

        // good each block has done its work and now we are left in last block
        tail();
        __syncthreads();
        if (0 == threadIdx.x) allDone = 1;
        __syncthreads();
      }

      // we need to wait the one above...
      while (0 == (allDone)) { __threadfence();}

      __threadfence();  // needed for whatever done in Tail?
      __syncthreads();

      assert(1==allDone);

  }


public:

  int32_t nWork;
  int32_t nDone;
  int32_t allDone;  // can be bool
};

#endif
