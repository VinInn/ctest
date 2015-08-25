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

}
