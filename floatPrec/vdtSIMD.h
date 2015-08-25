#pragma omp declare simd notinbranch
 double simd_asin(double x);
#pragma omp declare simd notinbranch
 float simd_asinf(float x);
#pragma omp declare simd notinbranch
 double simd_acos( double x );
#pragma omp declare simd notinbranch
 float simd_acosf( float x );
#pragma omp declare simd notinbranch
 double simd_atan(double x);
#pragma omp declare simd notinbranch
 float simd_atanf( float xx );
#pragma omp declare simd notinbranch
 float simd_atan2f( float y, float x );
#pragma omp declare simd notinbranch
 double simd_atan2( double y, double x );
#pragma omp declare simd notinbranch
 double simd_cos(double x);
#pragma omp declare simd notinbranch
 float simd_cosf(float x);
#pragma omp declare simd notinbranch
 double simd_exp(double initial_x);
#pragma omp declare simd notinbranch
 float simd_expf(float initial_x);
#pragma omp declare simd notinbranch
 double simd_inv_general(double x, const uint32_t isqrt_iterations);
#pragma omp declare simd notinbranch
 double simd_inv(double x);
#pragma omp declare simd notinbranch
 double simd_approx_inv(double x);
#pragma omp declare simd notinbranch
 float simd_invf_general(float x, const uint32_t isqrt_iterations);
#pragma omp declare simd notinbranch
 float simd_invf(float x);
#pragma omp declare simd notinbranch
 float simd_approx_invf(float x);
#pragma omp declare simd notinbranch
 double simd_log(double x);
#pragma omp declare simd notinbranch
 float simd_logf( float x );
#pragma omp declare simd notinbranch
 double simd_sin(double x);
#pragma omp declare simd notinbranch
 float simd_sinf(float x);
#pragma omp declare simd notinbranch
 void simd_sincos_m45_45( const double z, double & s, double &c );
#pragma omp declare simd notinbranch
 void simd_sincos( const double xx, double & s, double &c );
#pragma omp declare simd notinbranch
 void simd_sincosf_m45_45( const float x, float & s, float &c );
#pragma omp declare simd notinbranch
 void simd_sincosf( const float xx, float & s, float &c );
#pragma omp declare simd notinbranch
 double simd_isqrt_general(double x, const uint32_t ISQRT_ITERATIONS);
#pragma omp declare simd notinbranch
 double simd_isqrt(double x);
#pragma omp declare simd notinbranch
 double simd_approx_isqrt(double x);
#pragma omp declare simd notinbranch
 float simd_isqrtf_general(float x, const uint32_t ISQRT_ITERATIONS);
#pragma omp declare simd notinbranch
 float simd_isqrtf(float x);
#pragma omp declare simd notinbranch
 float simd_approx_isqrtf(float x);
#pragma omp declare simd notinbranch
 double simd_tan(double x);
#pragma omp declare simd notinbranch
 float simd_tanf(float x);
