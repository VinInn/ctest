#include<cassert>
#include<cstdint>
#include<cmath>
#include<random>
#include<vector>

#include "cudaCompat.h"


// #define VERIFY

#ifdef VERIFY
#define COUNT(x) atomicAdd(&counters[x],1);
#else
__device__
void dummy(int){}
#define COUNT(x) dummy(x);
#endif

template<int STRIDE, int NTTOT>
__global__
void nn(uint32_t * __restrict__ counters,
float const * __restrict__ z, float const * __restrict__ w, uint32_t * __restrict__ nns, int ntot, float eps) {
    COUNT(0);
    // this part is actually run blockDim.x times for each "z"
    auto id = blockIdx.y * blockDim.y + threadIdx.y;
    auto first = threadIdx.x;
    assert(blockDim.x==STRIDE);
    assert(blockIdx.x==0);
    assert(first<STRIDE);
    // usual loop uder the assumption ntot is not kown on HOST side
    auto incr = (blockDim.y * gridDim.y);
    for (auto j = id; j < ntot; j += incr) {
      COUNT(1)
      // combinatorial loop  (n^2)
      // in reality it should be limited using a Histogram, KDTree or similar
      // here we parallelize. for each "z[j]" blockDim.x threads are actually used
      auto k = j+ 1+first;
      for (;k < ntot; k +=blockDim.x) {
        COUNT(2);
        if (
             fabs(z[j]-z[k]) < eps && 
             fabs(w[j]-w[k]) < eps
           ) {
          atomicAdd(&nns[j],1);
          atomicAdd(&nns[k],1);
          COUNT(3);
        }
      }  // inner loop k
    } // outer loop j

}
