// rep -h fast_ ../../vdt/include/*.h | grep inline | sed 's/).*/);/g' | sed 's/fast/simd/' | sed 's/inline /#pragma omp declare simd notinbranch\
//  /' > vdtSIMD.h
// sed 's/\(.*\)_\(.*\)(\(.*\);/\1_\2(\3 { return fast_\2(x);}/' vdtSIMD.h > vdtSIMD.cc

#include "vdtMath.h"

namespace vdt {

  using namespace details;

#pragma omp declare simd notinbranch
 double simd_asin(double x) { return fast_asin(x);}
#pragma omp declare simd notinbranch
 float simd_asinf(float x) { return fast_asinf(x);}
#pragma omp declare simd notinbranch
 double simd_acos( double x ) { return fast_acos(x);}
#pragma omp declare simd notinbranch
 float simd_acosf( float x ) { return fast_acosf(x);}
#pragma omp declare simd notinbranch
 double simd_atan(double x) { return fast_atan(x);}
#pragma omp declare simd notinbranch
 float simd_atanf( float x ) { return fast_atanf(x);}
#pragma omp declare simd notinbranch
 float simd_atan2f( float y, float x ) { return fast_atan2f(y,x);}
#pragma omp declare simd notinbranch
 double simd_atan2( double y, double x ) { return fast_atan2(y,x);}
#pragma omp declare simd notinbranch
 double simd_cos(double x) { return fast_cos(x);}
#pragma omp declare simd notinbranch
 float simd_cosf(float x) { return fast_cosf(x);}
#pragma omp declare simd notinbranch
 double simd_exp(double x) { return fast_exp(x);}
#pragma omp declare simd notinbranch
 float simd_expf(float x) { return fast_expf(x);}
#pragma omp declare simd notinbranch uniform(isqrt_iterations)
 double simd_inv_general(double x, uint32_t isqrt_iterations) { return fast_inv_general(x,isqrt_iterations);}
#pragma omp declare simd notinbranch
 double simd_inv(double x) { return fast_inv(x);}
#pragma omp declare simd notinbranch
 double simd_approx_inv(double x) { return fast_inv(x);}
#pragma omp declare simd notinbranch uniform(isqrt_iterations)
 float simd_invf_general(float x, uint32_t isqrt_iterations) { return fast_invf_general(x,isqrt_iterations);}
#pragma omp declare simd notinbranch
 float simd_invf(float x) { return fast_invf(x);}
#pragma omp declare simd notinbranch
 float simd_approx_invf(float x) { return fast_invf(x);}
#pragma omp declare simd notinbranch
 double simd_log(double x) { return fast_log(x);}
#pragma omp declare simd notinbranch
 float simd_logf( float x ) { return fast_logf(x);}
#pragma omp declare simd notinbranch
 double simd_sin(double x) { return fast_sin(x);}
#pragma omp declare simd notinbranch
 float simd_sinf(float x) { return fast_sinf(x);}
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

