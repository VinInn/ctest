#include "hip/hip_runtime.h"

/* Search worst cases of a univariate binary32 function, by exhaustive search.

   This program is open-source software distributed under the terms 
   of the GNU General Public License <http://www.fsf.org/copyleft/gpl.html>.

   Compile with:

   $ gcc -DSTR=acos -O3 check_exhaustive.c -lmpfr -lgmp -lm -fopenmp
   $ icc -DSTR=acos -no-ftz -O3 check_exhaustive.c -lmpfr -lgmp -fopenmp
   $ nvcc -DSTR=acos check_exhaustive.cu --cudart shared -gencode arch=compute_70,code=sm_70 -O3 -std=c++17 --compiler-options="-O3 -lmpfr -lgmp -lm -fopenmp"

   By default it uses all threads available. To use for example 32 threads:

   $ OMP_NUM_THREADS=32 ./a.out

   For NEWLIB: add -DNEWLIB (to avoid compilation error with __errno).
*/

#ifndef STR
#error "please provide a value for STR"
#endif

typedef union { unsigned int n; float x; } union_t;


#ifdef __HIPCC__
#include<hip/hip_runtime.h>
#include<hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include<iostream>
inline
bool cudaCheck_(const char* file, int line, const char* cmd, hipError_t result)
{
    //std::cerr << file << ", line " << line << ": " << cmd << std::endl;
    if (result == hipSuccess)
        return true;

    const char* error = hipGetErrorName(result);
    const char* message = hipGetErrorString(result);
    std::cerr << file << ", line " << line << ": " << error << ": " << message << std::endl;
    abort();
    return false;
}
#define cudaCheck(ARG) (cudaCheck_(__FILE__, __LINE__, #ARG, (ARG)))
#endif

#ifndef __INTEL_COMPILER
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h> /* for ULONG_MAX */
#include <mpfr.h>
#include <assert.h>
#if !defined(__INTEL_COMPILER) && !defined(AMD) && !defined(MUSL)
#include <gnu/libc-version.h>
#endif
#include <omp.h>
#include <fenv.h>

#ifdef NEWLIB
/* RedHat's libm claims:
   undefined reference to `__errno' in j1f/y1f */
int errno;
int* __errno () { return &errno; }
#endif

/* https://stackoverflow.com/questions/1489932/how-to-concatenate-twice-with-the-c-preprocessor-and-expand-a-macro-as-in-arg */
#define FLOAT f
#define CAT1(X,Y) X ## Y
#define CAT2(X,Y) CAT1(X,Y)
#define FOO CAT2(STR,FLOAT)
#ifndef MPFR_FOO
#define MPFR_FOO CAT2(mpfr_,STR)
#endif
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define NAME TOSTRING(STR)

inline
int
mpfr_lgamma (mpfr_t y, mpfr_t x, mpfr_rnd_t r)
{
int s;
return mpfr_lgamma (y, &s, x, r);
}

inline
int
mpfr_tgamma (mpfr_t y, mpfr_t x, mpfr_rnd_t r)
{
return mpfr_gamma (y, x, r);
}


const int maxNumOfThreads = 256;
const int bunchSize = 1024;
float * ypD[maxNumOfThreads];
float * ypH[maxNumOfThreads];

#ifdef __HIPCC__
#include<hip/hip_runtime.h>
#include <hip/hip_runtime.h>

/*
__global__ void kernel_fsqrt(unsigned int n, float * py, int rd) {
  //__fsqrt_[rn,rz,ru,rd](x)
  auto fn = __fsqrt_rn;
  switch (rd) {
  case 0:
    fn = __fsqrt_rn; break;
  case 1:
    fn = __fsqrt_rz; break;
  case 2:
    fn = __fsqrt_ru; break;
  case 3:
    fn = __fsqrt_rd; break;
  default :
   fn = __fsqrt_rn; break;
  }
   int first = blockIdx.x * blockDim.x + threadIdx.x;
   for (int i=first; i<bunchSize; i+=gridDim.x*blockDim.x) {
     union_t u; u.n = n+i; float x = u.x;
     py[i] = fn(x);
  }
}
*/

hipStream_t streams[maxNumOfThreads];
__global__ void kernel_foo(unsigned int n, float * py) {
   int first = blockIdx.x * blockDim.x + threadIdx.x;
   for (int i=first; i<bunchSize; i+=gridDim.x*blockDim.x) {
     union_t u; u.n = n+i; float x = u.x;
     py[i] = FOO(x);
   }
}
#else // CPU version
void  kernel_foo(unsigned int n, float * py) 
#ifdef VECTORIZE
;
#else
{
   int first = 0;
   for (int i=first; i<bunchSize; i++) {
     union_t u;    u.n = n+i; float x = u.x;
     py[i] = FOO(x);
   }
}
#endif
#endif

float * wrap_foo(unsigned int n, int rd) {
  int nt = omp_get_thread_num();
#ifdef __HIPCC__
#ifdef DO_SQRT
  hipLaunchKernelGGL(kernel_fsqrt, dim3(1024/128), dim3(128), 0, streams[nt], n, ypD[nt],rd);
#else
  hipLaunchKernelGGL(kernel_foo, dim3(1024/128), dim3(128), 0, streams[nt], n, ypD[nt]);
#endif
  cudaCheck(hipMemcpyAsync(ypH[nt], ypD[nt], bunchSize*sizeof(float), hipMemcpyDeviceToHost, streams[nt]));
  hipStreamSynchronize(streams[nt]);
  // std::cout << "from GPU " << nt << ' ' << n << ' ' << ypH[nt][0] << std::endl;
#else
  kernel_foo(n, ypH[nt]);
  // std::cout << "from CPU "<< nt << ' ' << n << ' ' << ypH[nt][0] << std::endl;
#endif
  return ypH[nt];
}



int rnd1[] = { FE_TONEAREST, FE_TOWARDZERO, FE_UPWARD, FE_DOWNWARD };
mpfr_rnd_t rnd2[] = { MPFR_RNDN, MPFR_RNDZ, MPFR_RNDU, MPFR_RNDD };

mpfr_rnd_t rnd =  MPFR_RNDN;

static float
cr_foo (float x, int rnd)
{
  int inex;
  mpfr_t yy;
  float ret;
  mpfr_init2 (yy, 24);
  mpfr_set_flt (yy, x, MPFR_RNDN);
  inex = MPFR_FOO (yy, yy, rnd2[rnd]);
  mpfr_subnormalize (yy, inex, MPFR_RNDN);
  ret = mpfr_get_flt (yy, MPFR_RNDN);
  mpfr_clear (yy);
  return ret;
}

#if defined(__INTEL_COMPILER) || defined(MUSL)
int
isinff (float x)
{
  return isinf ((double) x);
}

int
isnanf (float x)
{
  return isnan ((double) x);
}
#endif

/* Return the error in ulps between y and z,
   where y is the result computed by libm (for input x),
   and z is the result computed by MPFR.
   Both y and z should not be NaN.
   Only one of y and z is allowed to be infinite. */
static unsigned long
ulp_error (float y, float z, float x)
{
  float err, ulp;
  if (y == z)
    return 0;
  if (isinff (z))
    {
      mpfr_t zz;
      if (isinff (y)) /* then y and z are of different signs */
      {
        assert (y * z < 0);
        return ULONG_MAX;
      }
      /* we divide everything by 2, taking as reference the MPFR function */
      y = y / 2;
      mpfr_init2 (zz, 24);
      mpfr_set_flt (zz, x, MPFR_RNDN);
      MPFR_FOO (zz, zz, rnd2[rnd]);
      // z = (float) (STR ((double) x) / 2);
      mpfr_div_2ui (zz, zz, 1, MPFR_RNDN);
      z = mpfr_get_flt (zz, MPFR_RNDN);
      /* If z is +Inf or -Inf, set it to +/-2^127 (since we divided y by 2) */
      if (isinff (z))
        z = (z > 0) ? 0x1p127 : -0x1p127;
      mpfr_clear (zz);
    }
  if (isinff (y))
    {
      assert (isinff (z) == 0);
      return ulp_error (z, y, x);
    }
  err = y - z;
  ulp = nextafterf (z, y) - z;
  err = fabsf (err / ulp);
  return (err >= (float) ULONG_MAX) ? ULONG_MAX : (unsigned long) err;
}

/* return the ulp error between y and FOO(x), where FOO(x) is computed with
   MPFR with 100 bits of precision */
static double
ulp_error_double (float y, float x)
{
  mpfr_t yy, zz;
  mpfr_prec_t prec = 100;
  int ret, inex;
  mpfr_exp_t e;
  double err;
  mpfr_exp_t emin = mpfr_get_emin ();
  mpfr_exp_t emax = mpfr_get_emax ();

  mpfr_set_emin (mpfr_get_emin_min ());
  mpfr_set_emax (mpfr_get_emax_max ());
  mpfr_init2 (yy, 24);
  mpfr_init2 (zz, prec);
  if (!isinff (y))
    {
      ret = mpfr_set_flt (yy, y, MPFR_RNDN);
      assert (ret == 0);
    }
  else
    mpfr_set_ui_2exp (yy, 1, 128, MPFR_RNDN);
  ret = mpfr_set_flt (zz, x, MPFR_RNDN);
  assert (ret == 0);
  inex = MPFR_FOO (zz, zz, MPFR_RNDN);
  e = mpfr_get_exp (zz);
  mpfr_sub (zz, zz, yy, MPFR_RNDA);
  mpfr_abs (zz, zz, MPFR_RNDN);
  /* we should add 2^(e - prec - 1) to |zz| */
  mpfr_set_ui_2exp (yy, 1, e - prec - 1, MPFR_RNDN);
  mpfr_add (zz, zz, yy, MPFR_RNDA);
  /* divide by ulp(y) */
  e = (e - 24 < -149) ? -149 : e - 24;
  mpfr_mul_2si (zz, zz, -e, MPFR_RNDN);
  err = mpfr_get_d (zz, MPFR_RNDA);
  mpfr_set_emin (emin);
  mpfr_set_emax (emax);
  mpfr_clear (yy);
  mpfr_clear (zz);
  return err;
}

#ifdef DEBUG
/* check if z is the correct rounding of x by computing another value
   with more precision and rounding it back */
static void
check_mpfr (float x, float z, int rnd)
{
  mpfr_t zz;
  mpfr_prec_t prec2 = 100;
  int inex;
  mpfr_init2 (zz, prec2);
  mpfr_set_flt (zz, x, MPFR_RNDN);
  MPFR_FOO (zz, zz, MPFR_RNDN);
  inex = mpfr_prec_round (zz, 24, rnd2[rnd]);
  mpfr_subnormalize (zz, inex, rnd2[rnd]);
  /* if inex=0, we can't conclude */
  if (inex != 0 && mpfr_get_flt (zz, MPFR_RNDN) != z)
    {
      fprintf (stderr, "Possible error in MPFR for x=%a\n", x);
      fprintf (stderr, "mpfr_%s (x) gives %a at precision 24\n", NAME, z);
      fprintf (stderr, "mpfr_%s (x) gives %a at precision 100\n", NAME,
               mpfr_get_flt (zz, MPFR_RNDN));
      fflush (stdout);
      exit (1);
    }
  mpfr_clear (zz);
}
#endif

unsigned long errors = 0;
unsigned long errors2 = 0; /* errors with 2 ulps or more */
unsigned long maxerr_u = 0;
unsigned int nmax = 0;
double maxerr = 0;

static void
check (unsigned int n, int rnd)
{

  fesetround (rnd1[rnd]);
  float * yp = wrap_foo(n,rnd);

  for (int i=0; i<bunchSize; ++i) {
  union_t u;
  float x, y, z;

  u.n = n+i;
  x = u.x;

  assert (!isnanf (x));
  assert (!isinff (x));

#ifdef EXCLUDE
  if (exclude (x))
    return;
#endif

  y = yp[i];
  z = cr_foo (x, rnd);

  if (y != z && !(isnanf (y) && isnanf (z)))
    {
      unsigned long err;
      double err_double;
#ifdef DEBUG
      if (!isinff (z))
        check_mpfr (x, z, rnd);
#endif
#pragma omp atomic update
      errors ++;
      err = ulp_error (y, z, x);
      if (err > 1)
#pragma	omp atomic update
        errors2 ++;
      err_double = ulp_error_double (y, x);
#pragma omp critical
      if (err > maxerr_u || (err == maxerr_u && err_double > maxerr))
        {
          maxerr_u = err;
          maxerr = (err_double > 0.5) ? err_double : 0.5;
          nmax = u.n;
        }
    }
  } // loop

}

static void
print_maximal_error (unsigned int n, int rnd)
{
  union_t u;
  float x, y, z;
  mpfr_t e;
  int ret;
  unsigned long err;
  double err_double;

  u.n = n;
  x = u.x;

  fesetround (rnd1[rnd]);
  float * yp = wrap_foo(n,rnd);
  y = yp[0];
  z = cr_foo (x, rnd);
  err = ulp_error (y, z, x);
  err_double = ulp_error_double (y, x);

  mpfr_init2 (e, 53);
  ret = mpfr_set_d (e, err_double, MPFR_RNDN);
  if (ret != 0)
    mpfr_printf ("x=%a y=%a z=%a err=%lu err_double=%a e=%Re\n", x, y, z, err, err_double, e);
  assert (ret == 0);
  mpfr_printf ("libm wrong by up to %.2RUe ulp(s) [%lu] for x=%a\n",
               e, err, x);
  printf ("%sf     gives %a\n", NAME, y);
  printf ("mpfr_%s gives %a\n", NAME, z);
  fflush (stdout);
  mpfr_clear (e);
}

int
main (int argc, char *argv[])
{

  int nstreams = omp_get_max_threads();
  assert(maxNumOfThreads>nstreams);

#ifdef __HIPCC__
  std::cout << "compiled with HIP " << std::endl;
  for (int i = 0; i < nstreams; i++)
    {
        cudaCheck(hipStreamCreate(&(streams[i])));
        cudaCheck(hipMalloc((void **)&ypD[i], bunchSize*sizeof(float)));
        cudaCheck(hipHostMalloc((void **)&ypH[i], bunchSize*sizeof(float)));

    }
#else
 for (int i = 0; i < nstreams; i++)
    {
      ypH[i] = (float *)malloc(bunchSize*sizeof(float));
    }
#endif

  int rnd = 0; /* default is rounding to nearest */

  if (argc >= 2)
    {
      if (strcmp (argv[1], "-rndn") == 0)
        {
          rnd = 0;
          argc --;
          argv ++;
        }
      else if (strcmp (argv[1], "-rndz") == 0)
        {
          rnd = 1;
          argc --;
          argv ++;
        }
      else if (strcmp (argv[1], "-rndu") == 0)
        {
          rnd = 2;
          argc --;
          argv ++;
        }
      else if (strcmp (argv[1], "-rndd") == 0)
        {
          rnd = 3;
          argc --;
          argv ++;
        }
      else
        {
          fprintf (stderr, "Error, unknown option %s\n", argv[1]);
          exit (1);
        }
    }

#ifdef __INTEL_COMPILER
  printf ("Using Intel Math Library\n");
#elif AMD
  printf ("Using AMD's library\n");
#elif NEWLIB
  printf ("Using RedHat newlib\n");
  // __fdlib_version = -1; /* __fdlibm_ieee */
#elif OPENLIBM
  printf ("Using OpenLibm\n");
#elif MUSL
  printf ("Using Musl\n");
#elif __HIPCC__
#ifndef CUDART_VERSION
 #warning "no " CUDART_VERSION
#endif
//  printf ("Using CUDA %d\n",CUDART_VERSION);
  int cuda_device = 0;
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, cuda_device);
  printf("CUDA Capable: SM %d.%d hardware\n", deviceProp.major, deviceProp.minor);
  
#else
  printf ("GNU libc version: %s\n", gnu_get_libc_version ());
  printf ("GNU libc release: %s\n", gnu_get_libc_release ());
#endif


  printf ("MPFR library: %-12s\nMPFR header:  %s (based on %d.%d.%d)\n",
          mpfr_get_version (), MPFR_VERSION_STRING, MPFR_VERSION_MAJOR,
          MPFR_VERSION_MINOR, MPFR_VERSION_PATCHLEVEL);
  printf ("Checking function %s with %s\n", NAME,
          mpfr_print_rnd_mode (rnd2[rnd]));
  fflush (stdout);

#define MAXN 2139095040U
#define INCR 1
#pragma omp parallel for schedule(dynamic,32)
  for (unsigned int n = 0; n < MAXN; n+=INCR*bunchSize)
    {
      /* we have to set emin/emax here, so that it is thread-local */
      mpfr_set_emin (-148);
      mpfr_set_emax (128);

      check (n, rnd);
      check (0x80000000 + n, rnd); /* negative values */
      if (0==n%1000000) std::cout << '.' << std::flush;
    }
  std::cout << std::endl;
  if (maxerr > 0.5)
    print_maximal_error (nmax, rnd);

  mpfr_t e;
  int ret;
  mpfr_init2 (e, 53);
  ret = mpfr_set_d (e, maxerr, MPFR_RNDN);
  assert (ret == 0);
  mpfr_printf ("Total: errors=%lu (%.2f%%) errors2=%lu maxerr=%.2RUe ulp(s)\n",
               errors, 100.0 * (double) errors / (double) (2 * MAXN),
               errors2, e);
  mpfr_clear (e);
  fflush (stdout);
  return 0;
}
