#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuFitVertices_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuFitVertices_h

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "cuda_assert.h"

#include "gpuVertexFinder.h"

namespace gpuVertexFinder {

  __device__ __forceinline__ void fitVertices(ZVertices* pdata,
                              WorkSpace* pws,
                              float chi2Max  // for outlier rejection
  ) {
    constexpr bool verbose = false;  // in principle the compiler should optmize out if false

    auto& __restrict__ data = *pdata;
    auto& __restrict__ ws = *pws;
    auto nt = ws.ntrks;
    float const* __restrict__ zt = ws.zt;
    float const* __restrict__ ezt2 = ws.ezt2;
    float const* __restrict__ tt = ws.tt;
    float const* __restrict__ ett2 = ws.ett2;
    float* __restrict__ zv = data.zv;
    float* __restrict__ wv = data.wv;
    float* __restrict__ tv = data.tv;
    float* __restrict__ wtv = data.wtv;
    float* __restrict__ chi2 = data.chi2;
    uint32_t& nvFinal = data.nvFinal;
    uint32_t& nvIntermediate = ws.nvIntermediate;

    int32_t* __restrict__ nn = data.ndof;
    int32_t* __restrict__ iv = ws.iv;

    assert(pdata);
    assert(zt);

    assert(nvFinal <= nvIntermediate);
    nvFinal = nvIntermediate;
    auto foundClusters = nvFinal;

    // zero
    for (auto i = threadIdx.x; i < foundClusters; i += blockDim.x) {
      zv[i] = 0;
      wv[i] = 0;
      tv[i] = 0;
      wtv[i] = 0;
      chi2[i] = 0;
      nn[i] = -2;  // ndof  (reuse it)
    }

    // only for test
    __shared__ int noise;
    if (verbose && 0 == threadIdx.x)
      noise = 0;

    __syncthreads();

    // compute cluster location
    for (auto i = threadIdx.x; i < nt; i += blockDim.x) {
      if (iv[i] > 9990) {
        if (verbose)
          atomicAdd(&noise, 1);
        continue;
      }
      assert(iv[i] >= 0);
      assert(iv[i] < int(foundClusters));
      auto wz = 1.f / ezt2[i];
      auto wt = 1.f / ett2[i];
      atomicAdd(&zv[iv[i]], zt[i] * wz);
      atomicAdd(&wv[iv[i]], wz);
      atomicAdd(&tv[iv[i]], tt[i] * wt);
      atomicAdd(&wtv[iv[i]], wt);
    }

    __syncthreads();
    for (auto i = threadIdx.x; i < foundClusters; i += blockDim.x) {
      if (wv[i] == 0.f) continue;
      zv[i] /= wv[i];
      tv[i] /= wtv[i];
    }
    __syncthreads();

    // compute chi2
    for (auto i = threadIdx.x; i < nt; i += blockDim.x) {
      if (iv[i] > 9990)
        continue;

      auto cz = zv[iv[i]] - zt[i];
      cz *= cz / ezt2[i];
      auto ct = tv[iv[i]] - tt[i];
      ct *= ct / ett2[i];
      auto c2 = cz+ct;
      if (c2 > 2.f*chi2Max) {
        iv[i] = 9999;
        continue;
      }
      atomicAdd(&chi2[iv[i]], c2);
      atomicAdd(&nn[iv[i]], 2);
    }
    __syncthreads();
    for (auto i = threadIdx.x; i < foundClusters; i += blockDim.x)
      if (nn[i] > 0)
        wv[i] *= float(nn[i]) / chi2[i];

    if (verbose && 0 == threadIdx.x)
      printf("found %d proto clusters ", foundClusters);
    if (verbose && 0 == threadIdx.x)
      printf("and %d noise\n", noise);
  }

  __global__ void fitVerticesKernel(ZVertices* pdata,
                              WorkSpace* pws,
                              float chi2Max  // for outlier rejection
  ) {

   fitVertices(pdata,pws,chi2Max);
 }

}  // namespace gpuVertexFinder

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuFitVertices_h
