#include "vdtMath.h"

namespace vdt {
  using namespace details;

#pragma omp declare simd notinbranch uniform(isqrt_iterations)
 double simd_inv_general(double x, uint32_t isqrt_iterations) { return fast_inv_general(x,isqrt_iterations);}
#pragma omp declare simd notinbranch
 double simd_inv(double x) { return fast_inv(x);}
#pragma omp declare simd notinbranch
 double simd_approx_inv(double x) { return fast_approx_inv(x);}
#pragma omp declare simd notinbranch uniform(isqrt_iterations)
 float simd_invf_general(float x, uint32_t isqrt_iterations) { return fast_invf_general(x,isqrt_iterations);}
#pragma omp declare simd notinbranch
 float simd_invf(float x) { return fast_invf(x);}
#pragma omp declare simd notinbranch
 float simd_approx_invf(float x) { return fast_approx_invf(x);}
#pragma omp declare simd notinbranch
 double simd_log(double x) { return fast_log(x);}
#pragma omp declare simd notinbranch
 float simd_logf( float x ) { return fast_logf(x);}
#pragma omp declare simd notinbranch
 double simd_sin(double x) { return fast_sin(x);}
#pragma omp declare simd notinbranch
 float simd_sinf(float x) { return fast_sinf(x);}

}

