#include "approx_exp.h"
#pragma omp declare simd notinbranch
template<int DEGREE>
float approx_vexpf(float x)
;
/* 
{
  return approx_expf<DEGREE>(x);
}
*/

#pragma omp declare simd notinbranch
template<>
float approx_vexpf<5>(float x)
{
  return approx_expf<5>(x);
}
#pragma omp declare simd notinbranch
template<>
float approx_vexpf<7>(float x)
{
  return approx_expf<7>(x);
}


float v0[1024];
float v1[1024];
float v2[1024];
float v3[1024];

void go() {
 #pragma omp simd
 for(int i=0; i<1024; ++i) {
   v0[i] = approx_vexpf<5>(v2[i]) + approx_vexpf<7>(v1[i]);
   v2[i] = approx_vexpf<7>(v0[i]) + approx_vexpf<5>(v1[i]);
   v3[i] = approx_vexpf<7>(v2[i]) + approx_vexpf<5>(v0[i]);

 }

}


