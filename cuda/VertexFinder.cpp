#include<random>
#include<vector>
#include<cstdint>
#include<cmath>

#include "cudaCompat.h"

#include "HistoContainer.h"

#include "VertexFinder.h"


#include<memory>
#include<iostream>

int main() {

  auto zt_d = std::make_unique<float[]>(64000);
  auto ezt2_d = std::make_unique<float[]>(64000);
  auto zv_d = std::make_unique<float[]>(256);
  auto wv_d = std::make_unique<float[]>(256);

  auto izt_d = std::make_unique<int8_t[]>(64000);
  auto nn_d = std::make_unique<uint16_t[]>(64000);
  auto iv_d = std::make_unique<int32_t[]>(64000);

  auto nv_d = std::make_unique<uint32_t[]>(1);
 
  auto onGPU_d = std::make_unique<OnGPU[]>(1);

  OnGPU onGPU;

  onGPU.zt = zt_d.get();
  onGPU.ezt2 = ezt2_d.get();
  onGPU.zv = zv_d.get();
  onGPU.wv = wv_d.get();
  onGPU.nv = nv_d.get();
  onGPU.izt = izt_d.get();
  onGPU.nn = nn_d.get();
  onGPU.iv = iv_d.get();


  memcpy(onGPU_d.get(), &onGPU, sizeof(OnGPU));


  Event  ev;

  for (int nav=30;nav<80;nav+=20){ 

  ClusterGenerator gen(nav,10);

  for (int i=4; i<20; ++i) {

  auto  kk=i/2;  // M param

  gen(ev);
  
  std::cout << ev.zvert.size() << ' ' << ev.ztrack.size() << std::endl;

  memcpy(onGPU.zt,ev.ztrack.data(),sizeof(float)*ev.ztrack.size());
  memcpy(onGPU.ezt2,ev.eztrack.data(),sizeof(float)*ev.eztrack.size());

  float eps = 0.1f;

  std::cout << "M eps " << kk << ' ' << eps << std::endl;

  clusterTracks(
		ev.ztrack.size(), onGPU_d.get(),kk,eps
		);


  uint32_t nv;
  assert(onGPU.nv);
  memcpy(&nv, onGPU.nv, sizeof(uint32_t));
  assert(nv>0); assert(nv<256);
  std::cout << "got "<< nv << " vertices" << std::endl;
  float zv[nv];
  float	wv[nv];
  memcpy(zv, onGPU.zv, nv*sizeof(float));
  memcpy(wv, onGPU.wv, nv*sizeof(float));

  float tw=0;
  for (auto w : wv) tw+=w;
  std::cout<< "total weight " << tw << std::endl;
  
  

  float dd[nv];
  uint32_t ii=0;
  for (auto zr : zv) {
   auto md=500.0f;
   for (auto zs : ev.ztrack) { 
     auto d = std::abs(zr-zs);
     md = std::min(d,md);
   }
   dd[ii++] = md;
  }
  assert(ii==nv);
  if (i==6) {
    for (auto d:dd) std::cout << d << ' ';
    std::cout << std::endl;
  }
  auto mx = std::minmax_element(dd,dd+nv);
  float rms=0;
  for (auto d:dd) rms+=d*d; rms = std::sqrt(rms)/(nv-1);
  std::cout << "min max rms " << *mx.first << ' ' << *mx.second << ' ' << rms << std::endl;

  } // loop on events
  } // lopp on ave vert
  
  return 0;
}
