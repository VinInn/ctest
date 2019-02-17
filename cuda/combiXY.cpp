#include "combiXY.h"

#include<iostream>
#include<memory>
#include<vector>


#include "cudaCompat.cc"

constexpr uint32_t NTOT = 1024*8;

template<int STRIDE, int NTTOT>
void go(uint32_t * c_d, float const * z_d, float const * w_d, uint32_t * nss_d) {
#ifdef VERIFY
  uint32_t counters[10];
  memset(c_d,0,10*sizeof(uint32_t));
#endif


  nn<STRIDE,NTTOT>(c_d, z_d,w_d,nss_d,NTOT,0.1f);

#ifdef VERIFY
  memcpy(counters,c_d,10*sizeof(uint32_t));

  std::cout << STRIDE << ' ' << NTOT;
  for (int i=0; i<5; ++i) std::cout << ' ' << counters[i];
  std::cout << std::endl;
#endif
}


int main() {


  auto z_d = std::make_unique<float[]>(NTOT);
  auto w_d = std::make_unique<float[]>(NTOT);
  auto nns_d = std::make_unique<uint32_t[]>(NTOT);
  auto c_d = std::make_unique<uint32_t[]>(10);

  for (int i=0; i<16; ++i) {

  memset(nns_d.get(),0,NTOT*sizeof(uint32_t));

  std::vector<float> z_h(NTOT);  // for "unclear" reasons this is now zeroed...
  std::mt19937 reng;
  std::uniform_real_distribution<float> rgen(-1.,1.);

  for (auto & z : z_h) z = rgen(reng);
  memcpy(z_d.get(),z_h.data(),sizeof(float)*z_h.size());
  for (auto & z : z_h) z = rgen(reng);
  memcpy(w_d.get(),z_h.data(),sizeof(float)*z_h.size());


  go<1,64>(c_d.get(), z_d.get(),w_d.get(),nns_d.get());

  go<1,256>(c_d.get(), z_d.get(),w_d.get(),nns_d.get());
 
 
  }

  return 0;

}
