#include "vdtMath.h"

namespace vdt {
  using namespace details;

#pragma omp declare simd notinbranch
 void simd_sincos_m45_45( const double z, double & s, double &c ) { return fast_sincos_m45_45(z,s,c);}
#pragma omp declare simd notinbranch
 void simd_sincos( const double x, double & s, double &c ) { return fast_sincos(x,s,c);}
#pragma omp declare simd notinbranch
 void simd_sincosf_m45_45( const float x, float & s, float &c ) { return fast_sincosf_m45_45(x,s,c);}
#pragma omp declare simd notinbranch
 void simd_sincosf( const float xx, float & s, float &c ) { return fast_sincosf(xx,s,c);}
#pragma omp declare simd notinbranch uniform(ISQRT_ITERATIONS)
 double simd_isqrt_general(double x, uint32_t ISQRT_ITERATIONS) { return fast_isqrt_general(x,ISQRT_ITERATIONS);}
#pragma omp declare simd notinbranch
 double simd_isqrt(double x) { return fast_isqrt(x);}
#pragma omp declare simd notinbranch
 double simd_approx_isqrt(double x) { return fast_isqrt(x);}
#pragma omp declare simd notinbranch uniform(ISQRT_ITERATIONS)
 float simd_isqrtf_general(float x, uint32_t ISQRT_ITERATIONS) { return fast_isqrtf_general(x,ISQRT_ITERATIONS);}
#pragma omp declare simd notinbranch
 float simd_isqrtf(float x) { return fast_isqrtf(x);}
#pragma omp declare simd notinbranch
 float simd_approx_isqrtf(float x) { return fast_isqrtf(x);}
#pragma omp declare simd notinbranch
 double simd_tan(double x) { return fast_tan(x);}
#pragma omp declare simd notinbranch
 float simd_tanf(float x) { return fast_tanf(x);}

}

