#include<cstdint>
#include<cmath>
#include<cassert>

#include "HistoContainer.h"

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


/*

__global__ 
void splitVertices(int nt,
                   OnGPU * pdata,
                   float maxChi2)  {
  
  
  
  auto & __restrict__ data = *pdata;
  float const * __restrict__ zt = data.zt;
  float const * __restrict__ ezt2 = data.ezt2;
  float * __restrict__ zv = data.zv;
  float * __restrict__ wv = data.wv;
  float * __restrict__ chi2 = data.chi2;
  uint32_t & nv = *data.nv;
  
  uint8_t  * __restrict__ izt = data.izt;
  int32_t * __restrict__ nn = data.nn;
  int32_t * __restrict__ iv = data.iv;
  
  assert(pdata);
  assert(zt);
  
  auto kv = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (kv>= nv) return;
  if (nn[kv]<4) return;
  if (chi2[kv]<maxChi2) return;
  
  assert(nn[kv]<1023];
  __shared__ float zz[1024];
  __shared__ uint_8 newV[1024];
  __shared__ float ww[1024];
  
  __shared__ int nq=0;
  for (auto k = threadIdx.x; k=nt; k+=blockDim.x) {
    if (iv[k]==kv) {
      auto old = atomicInc(nq,1024);
      zz[old] = zt[k]-zv[kv];
      newV[old] = zz[old]<0 ? 0 : 1;
      ww[old] = 1.f/ezt2[k];
    }
  }

  assert(nq=nn[kv]+1);
  
  __shared__ float znew[2], wnew[2];
  
  znew[0]=0; znew[1]=0;
  wnew[0]=0; wnew[1]=0;


  __syncthreads();

    int  maxiter=20;
  // kt-min....
  bool more = true;
  while(maxiter >0 && __syncthreads_or(more) ) {
    for (auto k = threadIdx.x; k=nq; k+=blockDim.x) {
      auto i = newV[k];
      atomicAdd(&znew[i],zz[k]*ww[k]);
      atomicAdd(&wnew[i],ww[k]);
    }
    __syncthreads();
    if(0==threadIdx.x) {
       znew[0]/=wnew[0];
       znew[1]/=wnew[1];
    }
    __syncthreads();
    for (auto k = threadIdx.x; k=nq; k+=blockDim.x) {
      d1 = fabs(zz[k]-znew[0]);
      d2 = fabs(zz[k]-znew[1]);
      auto newer = d1<d2 ? 0 : 1;
      more = newer = newV[k]
      __syncthreads();
 
    }
  }

}
  
*/


// this algo does not really scale as it works in a single block...
// enough for <10K tracks we have
__global__ 
void clusterTracks(int nt,
                   OnGPU * pdata,
                   int minT, float eps)  {

  float errmax = 0.02;  // max error to be "seed"
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
  __shared__ typename HistoContainer<int8_t,8,5,8,uint16_t>::Counter ws[32];
  for (auto k = threadIdx.x; k<hist.totbins(); k+=blockDim.x) hist.off[k]=0;
  __syncthreads();

//  if(0==threadIdx.x) printf("histo zeroed\n");


  // fill hist
  for (int i = threadIdx.x; i < nt; i += blockDim.x) {
    assert(i<64000);
    int iz =  int(zt[i]*10.);
    iz = std::max(iz,-127);
    iz = std::min(iz,127);
    izt[i]=iz;
    hist.count(int8_t(iz));
    iv[i]=i;
    nn[i]=0;
  }
  __syncthreads();
    if (threadIdx.x<32) ws[threadIdx.x]=0;  // used by prefix scan...
    __syncthreads();
    hist.finalize(ws);
    __syncthreads();
  
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

///  for test
struct Event {
  std::vector<float> zvert;
  std::vector<uint16_t>  itrack;
  std::vector<float> ztrack;
  std::vector<float> eztrack;
  std::vector<uint16_t> ivert;
};



struct ClusterGenerator {

  explicit ClusterGenerator(float nvert, float ntrack) :
    rgen(-13.,13), errgen(0.005,0.025), clusGen(nvert), trackGen(ntrack), gauss(0.,1.)
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
      auto err = 0.03f;
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
