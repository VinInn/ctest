#include<iostream>
#include<cassert>
#include<cstring>


#include <x86intrin.h>

unsigned int taux=0;
inline volatile unsigned long long rdtsc() {
 return __rdtscp(&taux);
}

void tof16(float const * f32, short * f16, unsigned int n) {
  assert(0==n%4);

  __m128 vf32;
  for (auto i=0U; i<n; i+=4) {
    ::memcpy(&vf32,f32+i,sizeof(vf32));
    auto vf16 = _mm_cvtps_ph (vf32,0) ;
    ::memcpy(f16+i,&vf16,sizeof(long long));
  }
			 
}

void tof32(short const * f16, float * f32, unsigned int n) {
  assert(0==n%4);

  __m128i vf16;
  for (auto i=0U; i<n; i+=4) {
    ::memcpy(&vf16,f16+i,sizeof(long long));
    auto vf32 = _mm_cvtph_ps (vf16) ;
    ::memcpy(f32+i, &vf32,sizeof(vf32));
  }
			 
}

inline
void tof16(float f32, short & f16) {

  __m128 vf32;                                                               
    ::memcpy(&vf32,&f32,sizeof(float));
    auto vf16 = _mm_cvtps_ph (vf32,0) ;
    ::memcpy(&f16,&vf16,sizeof(f16));
}


inline
void tof32(short f16, float & f32) {

  __m128i vf16;
   ::memcpy(&vf16,&f16,sizeof(f16));
    auto vf32 = _mm_cvtph_ps (vf16) ;
    ::memcpy(&f32, &vf32,sizeof(f32));

}

#include<algorithm>
int main() {

  float f[8]={3.14,-44.32,11.653,7.e-12,12.345678,-3456.71,4.21,-34.e7};
  short p[8];
  float r[8];

  tof16(f,p,8);
  tof32(p,r,8);
  
  for (auto x : r)  std::cout << x << ' ' ;
  std::cout<<std::endl;

  std::transform(f,f+8,p,[](auto x){ short y; tof16(x,y); return y;});
  std::transform(p,p+8,r,[](auto x){ float y; tof32(x,y); return y;});
  for (auto x : r)  std::cout << x << ' ' ;
  std::cout<<std::endl;
  

  return 0;
}

