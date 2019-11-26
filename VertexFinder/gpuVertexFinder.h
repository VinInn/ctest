#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuVertexFinder_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuVertexFinder_h

#include <cstddef>
#include <cstdint>

#include "ZVertexSoA.h"

namespace gpuVertexFinder {

  using ZVertices = ZVertexSoA;

  // workspace used in the vertex reco algos
  struct WorkSpace {
    static constexpr uint32_t MAXTRACKS = 16000;
    static constexpr uint32_t MAXVTX = 1024;

    uint32_t ntrks;            // number of "selected tracks"
    uint16_t itrk[MAXTRACKS];  // index of original track
    float zt[MAXTRACKS];       // input track z at bs
    float ezt2[MAXTRACKS];     // input error^2 on the above
    float ptt2[MAXTRACKS];     // input pt^2 on the above
    uint8_t izt[MAXTRACKS];    // interized z-position of input tracks
    int32_t iv[MAXTRACKS];     // vertex index for each associated track

    uint32_t nvIntermediate;  // the number of vertices after splitting pruning etc.

    __host__ __device__ void init() {
      ntrks = 0;
      nvIntermediate = 0;
    }
  };

  __global__ void init(ZVertexSoA* pdata, WorkSpace* pws) {
    pdata->init();
    pws->init();
  }

}  // namespace gpuVertexFinder

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuVertexFinder_h
