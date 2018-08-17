#include<random>
#include<vector>
#include<cstdint>
#include<cmath>

#include "HistoContainer.h"

#include <cuda/api_wrappers.h>


struct Event {
  std::vector<float> zvert;
  std::vector<uint16_t>  itrack;
  std::vector<float> ztrack;
  std::vector<float> eztrack;
  std::vector<uint16_t> ivert;
};


struct OnGPU {

  float * zt;
  float * ezt2;
  float * zv;
  float * wv;
  uint32_t * nv;
  int32_t * iv;

  // workspace  
  int8_t  * izt;
  uint16_t * nn;

};




// this algo does not really scale as it works in a single block...
// enough for <10K tracks we have
__global__ 
void clusterTracks(int nt,
                   OnGPU * pdata,
                   int minT, float eps)  {

  float errmax = 0.1;  // max error to be "seed"
  auto er2mx = errmax*errmax;

  auto & data = *pdata;
  float const * zt = data.zt;
  float const * ezt2 = data.ezt2;
  float * zv = data.zv;
  float * wv = data.wv;
  uint32_t & nv = *data.nv;

  int8_t  * izt = data.izt;
  uint16_t * nn = data.nn;
  int32_t * iv = data.iv;

  assert(pdata);
  assert(zt);

  __shared__ HistoContainer<int8_t,8,5,8,uint16_t> hist;

//  if(0==threadIdx.x) printf("params %d %f\n",minT,eps);    
//  if(0==threadIdx.x) printf("booked hist with %d bins, size %d for %d tracks\n",hist.nbins(),hist.binSize(),nt);

  // zero hist
  hist.nspills = 0;
  for (auto k = threadIdx.x; k<hist.nbins(); k+=blockDim.x) hist.n[k]=0;
  __syncthreads();

//  if(0==threadIdx.x) printf("histo zeroed\n");


  // fill hist
  for (int i = threadIdx.x; i < nt; i += blockDim.x) {
    assert(i<64000);
    int iz =  int(zt[i]*10.);
    iz = std::max(iz,-127);
    iz = std::min(iz,127);
    izt[i]=iz;
    hist.fill(int8_t(iz),uint16_t(i));
    iv[i]=i;
    nn[i]=0;
  }
  __syncthreads();

//   if(0==threadIdx.x) printf("histo filled %d\n",hist.nspills);
  if(0==threadIdx.x && hist.fullSpill()) printf("histo overflow\n");

  // count neighbours
  for (int i = threadIdx.x; i < nt; i += blockDim.x) {
     if (ezt2[i]>er2mx) continue;
     auto loop = [&](int j) {
        if (i==j) return;
        auto dist = std::abs(zt[i]-zt[j]);
        if (dist>eps) return;
        if (dist*dist>12.f*(ezt2[i]+ezt2[j])) return;
        nn[i]++;
     };

     int bs = hist.bin(izt[i]);
     int be = std::min(int(hist.nbins()),bs+2);
     bs = bs==0 ? 0 : bs-1;
     assert(be>bs);
     for (auto b=bs; b<be; ++b){
     for (auto pj=hist.begin(b);pj<hist.end(b);++pj) {
            loop(*pj);
     }}
     for (auto pj=hist.beginSpill();pj<hist.endSpill();++pj)
        loop(*pj);
  }

  __syncthreads();

//  if(0==threadIdx.x) printf("nn counted\n");

  // cluster seeds only
  bool more = true;
  while (__syncthreads_or(more)) {
    more=false;
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
      if (nn[i]<minT) continue; // DBSCAN core rule
      auto loop = [&](int j) {
        if (i==j) return;
        if (nn[j]<minT) return;  // DBSCAN core rule
        // look on the left
        auto dist = zt[j]-zt[i];
        if (dist<0 || dist>eps) return;
        if (dist*dist>12.f*(ezt2[i]+ezt2[j])) return;
        auto old = atomicMin(&iv[j], iv[i]);
        if (old != iv[i]) {
          // end the loop only if no changes were applied
          more = true;
        }
        atomicMin(&iv[i], old);
      };

      int bs = hist.bin(izt[i]);
      int be = std::min(int(hist.nbins()),bs+2);
      for (auto b=bs; b<be; ++b){
      for (auto pj=hist.begin(b);pj<hist.end(b);++pj) {
       	    loop(*pj);
      }}
      for (auto pj=hist.beginSpill();pj<hist.endSpill();++pj)
        loop(*pj);
    } // for i
  } // while



  // collect edges (assign to closest cluster of closest point??? here to closest point)
  for (int i = threadIdx.x; i < nt; i += blockDim.x) {
//    if (nn[i]==0 || nn[i]>=minT) continue;    // DBSCAN edge rule
    if (nn[i]>=minT) continue;    // DBSCAN edge rule
    float mdist=eps;
    auto loop = [&](int j) {
      if (nn[j]<minT) return;  // DBSCAN core rule
      auto dist = std::abs(zt[i]-zt[j]);
      if (dist>mdist) return;
      if (dist*dist>12.f*(ezt2[i]+ezt2[j])) return; // needed?
      mdist=dist;
      iv[i] = iv[j]; // assign to cluster (better be unique??)
    };
      int bs = hist.bin(izt[i]);
      int be = std::min(int(hist.nbins()),bs+2);
      bs = bs==0 ? 0 : bs-1;
      for (auto b=bs; b<be; ++b){
      for (auto pj=hist.begin(b);pj<hist.end(b);++pj) {
            loop(*pj);
      }}
      for (auto pj=hist.beginSpill();pj<hist.endSpill();++pj)
        loop(*pj);
   }
   

    __shared__ int foundClusters;
    foundClusters = 0;
    __syncthreads();

    // find the number of different clusters, identified by a pixels with clus[i] == i;
    // mark these pixels with a negative id.
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
        if (iv[i] == i) {
          if  (nn[i]>=minT) {
            auto old = atomicAdd(&foundClusters, 1);
            iv[i] = -(old + 1);
            zv[old]=0;
            wv[old]=0;
          } else { // noise
           iv[i] = -9998;
          }
       }
    }
    __syncthreads();

    // propagate the negative id to all the pixels in the cluster.
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
        if (iv[i] >= 0) {
          // mark each pixel in a cluster with the same id as the first one
          iv[i] = iv[iv[i]];
        }
    }
    __syncthreads();

    // adjust the cluster id to be a positive value starting from 0
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
        iv[i] = - iv[i] - 1;
    }

    __shared__ int noise;
    noise = 0;

    __syncthreads();

    // compute cluster location
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
      if (iv[i]>9990) { atomicAdd(&noise, 1); continue;}
      assert(iv[i]>=0);
      assert(iv[i]<foundClusters);
      auto w = 1.f/ezt2[i];
      atomicAdd(&zv[iv[i]],zt[i]*w);
      atomicAdd(&wv[iv[i]],w); 
    }

    __syncthreads();

   if(0==threadIdx.x) printf("found %d proto clusters ",foundClusters);
   if(0==threadIdx.x) printf("and %d noise\n",noise);

   for (int i = threadIdx.x; i < foundClusters; i += blockDim.x) zv[i]/=wv[i];

   nv = foundClusters;
}


struct ClusterGenerator {

  explicit ClusterGenerator(float nvert, float ntrack) :
    rgen(-13.,13), errgen(0.007,0.04), clusGen(nvert), trackGen(ntrack), gauss(0.,1.)
  {}

  void operator()(Event & ev) {

    int nclus = clusGen(reng);
    ev.zvert.resize(nclus);
    ev.itrack.resize(nclus);
    for (auto & z : ev.zvert) { 
       z = 3.5f*gauss(reng);
    }

    ev.ztrack.clear(); 
    ev.eztrack.clear();
    ev.ivert.clear();
    for (int iv=0; iv<nclus; ++iv) {
      auto nt = trackGen(reng);
      ev.itrack[nclus] = nt;
      for (int it=0; it<nt; ++it) {
       auto err = errgen(reng); // reality is not flat....
       ev.ztrack.push_back(ev.zvert[iv]+err*gauss(reng));
       ev.eztrack.push_back(err*err);
       ev.ivert.push_back(iv);
      }
    }
    // add noise
    auto nt = 2*trackGen(reng);
    for (int it=0; it<nt; ++it) {
      auto err = 0.05f;
      ev.ztrack.push_back(rgen(reng));
      ev.eztrack.push_back(err*err);
      ev.ivert.push_back(9999);
    }

  }

  std::mt19937 reng;
  std::uniform_real_distribution<float> rgen;
  std::uniform_real_distribution<float> errgen;
  std::poisson_distribution<int> clusGen;
  std::poisson_distribution<int> trackGen;
  std::normal_distribution<float> gauss;


};


#include<iostream>

int main() {

  if (cuda::device::count() == 0) {
    std::cerr << "No CUDA devices on this system" << "\n";
    exit(EXIT_FAILURE);
  }

  auto current_device = cuda::device::current::get();

  auto zt_d = cuda::memory::device::make_unique<float[]>(current_device, 64000);
  auto ezt2_d = cuda::memory::device::make_unique<float[]>(current_device, 64000);
  auto zv_d = cuda::memory::device::make_unique<float[]>(current_device, 256);
  auto wv_d = cuda::memory::device::make_unique<float[]>(current_device, 256);

  auto izt_d = cuda::memory::device::make_unique<int8_t[]>(current_device, 64000);
  auto nn_d = cuda::memory::device::make_unique<uint16_t[]>(current_device, 64000);
  auto iv_d = cuda::memory::device::make_unique<int32_t[]>(current_device, 64000);

  auto nv_d = cuda::memory::device::make_unique<uint32_t[]>(current_device, 1);
 
  auto onGPU_d = cuda::memory::device::make_unique<OnGPU[]>(current_device, 1);

  OnGPU onGPU;

  onGPU.zt = zt_d.get();
  onGPU.ezt2 = ezt2_d.get();
  onGPU.zv = zv_d.get();
  onGPU.wv = wv_d.get();
  onGPU.nv = nv_d.get();
  onGPU.izt = izt_d.get();
  onGPU.nn = nn_d.get();
  onGPU.iv = iv_d.get();


  cuda::memory::copy(onGPU_d.get(), &onGPU, sizeof(OnGPU));


  Event  ev;

  for (int nav=30;nav<80;nav+=20){ 

  ClusterGenerator gen(nav,10);

  for (int i=4; i<20; ++i) {

  auto  kk=i/2;  // M param

  gen(ev);
  
  std::cout << ev.zvert.size() << ' ' << ev.ztrack.size() << std::endl;

  cuda::memory::copy(onGPU.zt,ev.ztrack.data(),sizeof(float)*ev.ztrack.size());
  cuda::memory::copy(onGPU.ezt2,ev.eztrack.data(),sizeof(float)*ev.eztrack.size());

  float eps = 0.1f;

  std::cout << "M eps " << kk << ' ' << eps << std::endl;

  cuda::launch(clusterTracks,
                { 1, 1024 },
                ev.ztrack.size(), onGPU_d.get(),kk,eps
           );


  uint32_t nv;
  cuda::memory::copy(&nv, onGPU.nv, sizeof(uint32_t));
  float zv[nv];
  float	wv[nv];
  cuda::memory::copy(&zv, onGPU.zv, nv*sizeof(float));
  cuda::memory::copy(&wv, onGPU.wv, nv*sizeof(float));

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
