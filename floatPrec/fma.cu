// nvcc -gencode arch=compute_61,code=sm_61 -fmad=false -O3 -ptx fma.cu -o fma.ptx -I/cvmfs/cms.cern.ch/slc7_amd64_gcc630/external/cuda/9.1.85-cms/include ; cat fma.ptx
// c++ -O3 -S -march=native -ffp-contract=off fma.cc ; cat fma.s


#include <cmath>

#ifdef __NVCC__
#define inline __device__  __host__ inline
#else
#define __global__
#endif

#if defined(__x86_64__) && !defined(__FMA__)
#warning nofma
#define FMA(x,y,z) x*y+z
#else
#warning fma
#define	FMA(x,y,z) std::fma(x,y,z)
#endif

inline
float dofma(float x, float y, float z) {
  return FMA(x,-y,z);
}


inline
float myf(float x, float y, float z) {
  return std::fma(x,-y,z);
}

inline
float myff(float x, float y, float z) {
  return z+x*y;
}

inline
float myfn(float x, float y, float z) {
  return x*y-z;
}


inline
float myxyn(float x, float y, float z) {
  return (x*y) - (y*z);
}

inline
float myxyp(float x, float y, float z) {
  return (x*y) + (y*z);
}

inline
float logP(float y) {
  return  y * (float(0xf.fff14p-4) + y * (-float(0x7.ff4bfp-4) 
  + y * (float(0x5.582f6p-4) + y * (-float(0x4.1dcf2p-4) + y * (float(0x3.3863f8p-4) + y * (-float(0x1.9288d4p-4)))))));

}

inline
float cw(float x) {
  constexpr float inv_log2f = float(0x1.715476p0);
  constexpr float log2H = float(0xb.172p-4);
  constexpr float log2L = float(0x1.7f7d1cp-20);
  // This is doing round(x*inv_log2f) to the nearest integer
  // float z = std::round(x*inv_log2f);
  float z = std::floor((x*inv_log2f) +0.5f);
  float y;
  // Cody-and-Waite accurate range reduction. FMA-safe.
  y = x;
  y -= z*log2H;
  y -= z*log2L;
  return y;
}




__global__
void goGPU(float * x, float * y, float * z, float * r) {

  r[9] = dofma(x[9],y[9],z[9]);

  r[0] = myf(x[0],y[0],z[0]);

  r[1] = myff(x[1],y[1],z[1]);

  r[2] = myfn(x[2],y[2],z[2]);

  r[3] = myxyn(x[3],y[3],z[3]);

  r[4] = myxyp(x[4],y[4],z[4]);

  r[5] = logP(x[5]);

  r[6] = cw(x[6]);



}


