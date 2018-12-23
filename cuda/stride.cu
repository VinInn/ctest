#include<cassert>
#include<cstdint>
#include<cmath>
#include<random>
#include<vector>

template<int STRIDE>
__global__
void nn(float const * __restrict__ z, float const * __restrict__ w, uint32_t * __restrict__ nns, int ntot, float eps) {
    // this part is actually run STRIDE times for each "z"
    auto ldx = blockIdx.x * blockDim.x + threadIdx.x;
    auto idx = ldx/STRIDE;
    auto first = ldx - idx*STRIDE;
    assert(first<STRIDE);
    // usual loop uder the assumption ntot is not kown on HOST side
    auto incr = (blockDim.x * gridDim.x)/STRIDE;
    for (auto j = idx; j < ntot; j += incr) {

      // combinatorial loop  (n^2)
      // in reality it should be limited using a Histogram, KDTree or similar
      // here we parallelize. for each "z[j]" STRIDE threads are actually used
      auto k = j+ 1+first;
      for (;k < ntot; k +=STRIDE) {
        if (
             fabs(z[j]-z[k]) < eps && 
             fabs(w[j]-w[k]) < eps
           ) {
          atomicAdd(&nns[j],1);
          atomicAdd(&nns[k],1);
        }
      }  // inner loop k
    } // outer loop j

}

#include <cuda/api_wrappers.h>
#include<iostream>


constexpr uint32_t NTOT = 1024*8;

template<int STRIDE>
void go(float * z_d, float * w_d, uint32_t * nss_d) {

  auto nt = 128;
  auto nb = 1024*STRIDE;

  nn<STRIDE><<<nb,nt>>>(z_d,w_d,nss_d,NTOT,0.1f);

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

  for (int i=0; i<16; ++i) {

  cudaMemset(nns_d.get(),0,NTOT*sizeof(uint32_t));

  std::vector<float> z_h(NTOT);  // for "unclear" reasons this is now zeroed...
  std::mt19937 reng;
  std::uniform_real_distribution<float> rgen(-1.,1.);

  for (auto & z : z_h) z = rgen(reng);
  cuda::memory::copy(z_d.get(),z_h.data(),sizeof(float)*z_h.size());
  for (auto & z : z_h) z = rgen(reng);
  cuda::memory::copy(w_d.get(),z_h.data(),sizeof(float)*z_h.size());


  go<1>(z_d.get(),w_d.get(),nns_d.get());
  go<2>(z_d.get(),w_d.get(),nns_d.get());
  go<4>(z_d.get(),w_d.get(),nns_d.get());
  go<8>(z_d.get(),w_d.get(),nns_d.get());
  go<16>(z_d.get(),w_d.get(),nns_d.get());


  }
  cudaDeviceSynchronize();

  return 0;

}
