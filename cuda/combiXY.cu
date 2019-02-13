#include "combiXY.h"


#include <cuda/api_wrappers.h>
#include<iostream>


constexpr uint32_t NTOT = 1024*8;

template<int STRIDE, int NTTOT>
void go(uint32_t * c_d, float const * z_d, float const * w_d, uint32_t * nss_d) {
#ifdef VERIFY
  uint32_t counters[10];
  cudaMemset(c_d,0,10*sizeof(uint32_t));
#endif

  // x is the fastest....
  auto ntTot = NTTOT; // KEEP THE NUMBER OF THREAD FIXED
  auto nty = ntTot/STRIDE;
  auto nby = 1024*STRIDE/NTTOT*64; // 1024 was for ntTot=64
  auto ntx = STRIDE;
  auto nbx = 1;
  dim3 nt(ntx,nty,1);
  dim3 nb(nbx,nby,1);
  nn<STRIDE,NTTOT><<<nb,nt>>>(c_d, z_d,w_d,nss_d,NTOT,0.1f);

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


  go<1,64>(c_d.get(), z_d.get(),w_d.get(),nns_d.get());
  go<2,64>(c_d.get(),z_d.get(),w_d.get(),nns_d.get());
  go<4,64>(c_d.get(),z_d.get(),w_d.get(),nns_d.get());
  go<8,64>(c_d.get(),z_d.get(),w_d.get(),nns_d.get());
  go<16,64>(c_d.get(),z_d.get(),w_d.get(),nns_d.get());
  go<32,64>(c_d.get(),z_d.get(),w_d.get(),nns_d.get());

  go<1,256>(c_d.get(), z_d.get(),w_d.get(),nns_d.get());
  go<2,256>(c_d.get(),z_d.get(),w_d.get(),nns_d.get());
  go<4,256>(c_d.get(),z_d.get(),w_d.get(),nns_d.get());
  go<8,256>(c_d.get(),z_d.get(),w_d.get(),nns_d.get());
  go<16,256>(c_d.get(),z_d.get(),w_d.get(),nns_d.get());
  go<32,256>(c_d.get(),z_d.get(),w_d.get(),nns_d.get());
  go<64,256>(c_d.get(),z_d.get(),w_d.get(),nns_d.get());

  go<64,1024>(c_d.get(),z_d.get(),w_d.get(),nns_d.get());
  go<128,1024>(c_d.get(),z_d.get(),w_d.get(),nns_d.get());
  go<256,1024>(c_d.get(),z_d.get(),w_d.get(),nns_d.get());

  }
  cudaDeviceSynchronize();

  return 0;

}
