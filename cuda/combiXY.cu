#include<cassert>
#include<cstdint>
#include<cmath>
#include<random>
#include<vector>

// #define VERIFY

#ifdef VERIFY
#define COUNT(x) atomicAdd(&counters[x],1);
#else
__device__
void dummy(int){}
#define COUNT(x) dummy(x);
#endif

template<int STRIDE>
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

#include <cuda/api_wrappers.h>
#include<iostream>


constexpr uint32_t NTOT = 1024*8;

template<int STRIDE>
void go(uint32_t * c_d, float const * z_d, float const * w_d, uint32_t * nss_d) {
#ifdef VERIFY
  uint32_t counters[10];
  cudaMemset(c_d,0,10*sizeof(uint32_t));
#endif

  // x is the fastest....
  auto nty = 64;
  auto nby = 1024;
  auto ntx = STRIDE;
  auto nbx = 1;
  dim3 nt(ntx,nty,1);
  dim3 nb(nbx,nby,1);
  nn<STRIDE><<<nb,nt>>>(c_d, z_d,w_d,nss_d,NTOT,0.1f);

#ifdef VERIFY
  cuda::memory::copy(counters,c_d,10*sizeof(uint32_t));

  std::cout << STRIDE << ' ' << NTOT;
  for (int i=0; i<5; ++i) std::cout << ' ' << counters[i];
  std::cout << std::endl;
#endif
}


int main() {

  if (cuda::device::count() == 0) {
    std::cerr << "No CUDA devices on this system" << "\n";
    exit(EXIT_FAILURE);
  }

  auto current_device = cuda::device::current::get();

  auto z_d = cuda::memory::device::make_unique<float[]>(current_device, NTOT);
  auto w_d = cuda::memory::device::make_unique<float[]>(current_device, NTOT);
  auto nns_d = cuda::memory::device::make_unique<uint32_t[]>(current_device, NTOT);
  auto c_d = cuda::memory::device::make_unique<uint32_t[]>(current_device, 10);

  for (int i=0; i<16; ++i) {

  cudaMemset(nns_d.get(),0,NTOT*sizeof(uint32_t));

  std::vector<float> z_h(NTOT);  // for "unclear" reasons this is now zeroed...
  std::mt19937 reng;
  std::uniform_real_distribution<float> rgen(-1.,1.);

  for (auto & z : z_h) z = rgen(reng);
  cuda::memory::copy(z_d.get(),z_h.data(),sizeof(float)*z_h.size());
  for (auto & z : z_h) z = rgen(reng);
  cuda::memory::copy(w_d.get(),z_h.data(),sizeof(float)*z_h.size());


  go<1>(c_d.get(), z_d.get(),w_d.get(),nns_d.get());
  go<2>(c_d.get(),z_d.get(),w_d.get(),nns_d.get());
  go<4>(c_d.get(),z_d.get(),w_d.get(),nns_d.get());
  go<8>(c_d.get(),z_d.get(),w_d.get(),nns_d.get());
  go<16>(c_d.get(),z_d.get(),w_d.get(),nns_d.get());


  }
  cudaDeviceSynchronize();

  return 0;

}
