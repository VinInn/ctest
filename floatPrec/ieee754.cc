#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cstring>

  union binary32 {
    binary32() : ui32(0) {};
    binary32(float ff) : f(ff) {};
    binary32(int32_t ii) : i32(ii){}
    binary32(uint32_t ui) : ui32(ui){}
    
    uint32_t ui32; /* unsigned int */                
    int32_t i32; /* Signed int */                
    float f;
  };




inline
float ilog(float x) {

  constexpr float inv_log2f = float(0x1.715476p0);
  float z = std::floor((x*inv_log2f) +0.5f);
  constexpr float log2F = 0xb.17218p-4;
  float y=x-z*log2F;
  // exponent 
  int32_t e = z;
  

  uint32_t biased_exponent= e+127;

#ifdef MEMCPY
  uint32_t ei = biased_exponent<<23;
  float ef;
  memcpy(&ef,&ei,sizeof(float));
  return ef;
#else
 binary32 ef;
  ef.ui32=(biased_exponent<<23);
  return ef.f;
#endif

}


alignas(32) float a[1024], b[1024];

void bar() {
#pragma omp simd
  for (int i=0; i<1024; ++i)
      b[i]=ilog(a[i]);
}


void foo(float const * x, float * y, int N) {
#pragma omp simd aligned(x, y: 32)
  for (int i=0; i<N; ++i)
      y[i]=ilog(x[i]);
}


float sum(float const * x, float * y, int N) {
   float s=0;
#pragma omp simd aligned(x, y: 32) reduction(+: s)
  for (int i=0; i<N; ++i)
      s+=y[i]+ilog(x[i]);
  return s;
}

