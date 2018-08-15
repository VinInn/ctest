#include<random>
#include<vector>
#include<cstdint>

#include "HistoContainer.h"


struct Event {
  std::vector<float> zvert;
  std::vector<uint16_t>  itrack;
  std::vector<float> ztrack;
  std::vector<uint16_t> ivert;
};


struct OnGPU {

  float * zt;
  float * zv;
  uint32_t * nv;
  
  int8_t  * izt;
  uint16_t * nn;
  uint32_t * iv;
};


// this algo does not really scale as it works in a single block...
// enough for <10K tracks we have
__global__ 
void clusterTracks(int nt, float const * zt, int8_t  * izt, uint16_t * nn, int32_t * iv, float * zv, uint32_t * nv, int minT, float eps)  {

  HistoContainer<int8_t,8,4,8,uint16_t> hist;

  // zero hist
  hist.nspills = 0;
  for (auto k = threadIdx.x; k<hist.nbins(); k+=blockDim.x) hist.n[k]=0;
  __syncthreads();


  // fill hist
  for (int i = threadIdx.x; i < nt; i += blockDim.x) {
    int iz =  int(zt[i]*10.);
    iz = std::max(iz,-127);
    iz = std::min(iz,127);
    izt[i]=iz;
    hist.fill(izt,i);
    iv[i]=i;
    nn[i]=0;
  }
  __syncthreads();

  // count neighbours
  for (int i = threadIdx.x; i < nt; i += blockDim.x) {

     auto loop = [&](int j) {
        if (i==j) return;
        auto dist = std::abs(zt[i]-zt[j]);
        if (dist<eps) return;
        nn[i]++;
     };

     auto bs = hist.bin(izt[i]);
     auto be = std::min(hist.nbins(),bs+2);
     bs = bs==0 ? 0 : bs-1;
     for (auto b=bs; b<be; ++b){
     for (auto pj=hist.begin(b);pj<hist.end(b);++pj) {
            loop(*pj);
     }}
     for (auto pj=hist.beginSpill();pj<hist.endSpill();++pj)
        loop(*pj);
  }

  __syncthreads();


  bool done = false;
  while (not __syncthreads_and(done)) {
    done = true;
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {

      auto loop = [&](int j) {
       if (i>=j) return;
       if (nn[i]<minT && nn[j]<minT) return;  // DBSCAN rule
        auto dist = std::abs(zt[i]-zt[j]);
        if (dist<eps) return;
        auto old = atomicMin(&iv[j], iv[i]);
        if (old != iv[i]) {
          // end the loop only if no changes were applied
          done = false;
        }
        atomicMin(&iv[i], old);
      }; 
      auto bs = hist.bin(izt[i]);
      auto be = std::min(hist.nbins(),bs+2);
      bs = bs==0 ? 0 : bs-1;
      for (auto b=bs; b<be; ++b){
      for (auto pj=hist.begin(b);pj<hist.end(b);++pj) {
       	    loop(*pj);
      }}
      for (auto pj=hist.beginSpill();pj<hist.endSpill();++pj)
        loop(*pj);
    } // for i
  } // while


    __shared__ int foundClusters;
    foundClusters = 0;
    __syncthreads();

    // find the number of different clusters, identified by a pixels with clus[i] == i;
    // mark these pixels with a negative id.
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
        if (iv[i] == i) {
          auto old = atomicAdd(&foundClusters, 1);
          iv[i] = -(old + 1);
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
    __syncthreads();


}


struct ClusterGenerator {

  explicit ClusterGenerator(float nvert, float ntrack) :
    rgen(-13.,13), clusGen(nvert), trackGen(ntrack), gauss(0.,1.)
  {}

  void operator()(Event & ev) {

    int nclus = clusGen(reng);
    ev.zvert.resize(nclus);
    ev.itrack.resize(nclus);
    for (auto & z : ev.zvert) { 
       z = 13.0f*gauss(reng);
    }

    ev.ztrack.clear(); 
    ev.ivert.clear();
    for (int iv=0; iv<nclus; ++iv) {
      auto nt = trackGen(reng);
      ev.itrack[nclus] = nt;
      for (int it=0; it<nt; ++it) {
       ev.ztrack.push_back(ev.zvert[iv]+0.02f*gauss(reng));  // reality is not gaussian....
       ev.ivert.push_back(iv);
      }
    }
    // add noise
    auto nt = 2*trackGen(reng);
    for (int it=0; it<nt; ++it) {
      ev.ztrack.push_back(rgen(reng));
      ev.ivert.push_back(9999);
    }

  }

  std::mt19937 reng;
  std::uniform_real_distribution<float> rgen;
  std::poisson_distribution<int> clusGen;
  std::poisson_distribution<int> trackGen;
  std::normal_distribution<float> gauss;


};


#include<iostream>

int main() {


  Event  ev;

  ClusterGenerator gen(50,10);

  gen(ev);
  
  std::cout << ev.zvert.size() << ' ' << ev.ztrack.size() << std::endl;

}
