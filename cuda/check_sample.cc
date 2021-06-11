/* Search worst cases of a univariate function, using a recursive algorithm.

   This program is open-source software distributed under the terms 
   of the GNU General Public License <http://www.fsf.org/copyleft/gpl.html>.

   Compile with:

   gcc -DFOO=acos -DUSE_xxx -O3 check_sample.c -lmpfr -lgmp -lm -fopenmp
   icc -DFOO=acos -Qoption,cpp,--extended_float_types -no-ftz -DUSE_xxx -O3 check_sample.c -lmpfr -lgmp -fopenmp

   where xxx is FLOAT, DOUBLE, LDOUBLE, or FLOAT128.

   For NEWLIB: add -DNEWLIB (to avoid compilation error with __errno).

   You can add -DWORST to use some precomputed values to guide the search.

   You can add -DGLIBC to print the GNU libc release (with -v).

   References and credit:
   * https://www.vinc17.net/research/testlibm/: worst-cases computed by
     Vincent Lefèvre.
   * the idea to sample several intervals instead of only one is due to
     Eric Schneider
*/

#ifndef FOO
#error "please provide a value for FOO"
#endif


#ifndef _GNU_SOURCE
#define _GNU_SOURCE /* to define ...f128 functions */
#endif


#define RANK /* print the maximal list-rank of best values */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#ifndef NO_FLOAT128
#define MPFR_WANT_FLOAT128
#endif
#include <mpfr.h>
#include <math.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>
#ifndef NO_OPENMP
#include <omp.h>
#endif
#include <float.h> /* for DBL_MAX */
#include <fenv.h>

/* define GLIBC to print the GNU libc version */
#ifdef GLIBC
#include <gnu/libc-version.h>
#endif


#ifdef __CUDACC__
#include<cuda.h>
#include<cuda_runtime.h>
#include <cuda_runtime_api.h>
#include<iostream>
inline
bool cudaCheck_(const char* file, int line, const char* cmd, CUresult result)
{
    //std::cerr << file << ", line " << line << ": " << cmd << std::endl;
    if (result == CUDA_SUCCESS)
        return true;

    const char* error;
    const char* message;
    cuGetErrorName(result, &error);
    cuGetErrorString(result, &message);
//    std::cerr << file << ", line " << line << ": " << error << ": " << message << std::endl;
    abort();
    return false;
}

inline
bool cudaCheck_(const char* file, int line, const char* cmd, cudaError_t result)
{
    //std::cerr << file << ", line " << line << ": " << cmd << std::endl;
    if (result == cudaSuccess)
        return true;

    const char* error = cudaGetErrorName(result);
    const char* message = cudaGetErrorString(result);
    std::cerr << file << ", line " << line << ": " << error << ": " << message << std::endl;
    abort();
    return false;
}
#define cudaCheck(ARG) (cudaCheck_(__FILE__, __LINE__, #ARG, (ARG)))
#endif



/* rounding modes */
int rnd1[] = { FE_TONEAREST, FE_TOWARDZERO, FE_UPWARD, FE_DOWNWARD };
mpfr_rnd_t rnd2[] = { MPFR_RNDN, MPFR_RNDZ, MPFR_RNDU, MPFR_RNDD };
mpfr_rnd_t rnd = MPFR_RNDN; /* default rounding mode */

/* mode (0,1,2), if -1 set according to omp_get_thread_num() */
int use_mode = -1;

#ifdef NEWLIB
/* without this, newlib says: undefined reference to `__errno' */
int errno;
int* __errno () { return &errno; }

/* cf https://sourceware.org/pipermail/newlib/2020/018027.html */
float
mylgammaf (float x)
{
  int s;
  return lgammaf_r (x, &s);
}

double
mylgamma (double x)
{
  int s;
  return lgamma_r (x, &s);
}
#endif

#define CAT1(X,Y) X ## Y
#define CAT2(X,Y) CAT1(X,Y)
#ifndef MPFR_FOO
#define MPFR_FOO CAT2(mpfr_,FOO)
#endif
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define NAME TOSTRING(FOO)

#ifdef USE_FLOAT
#if defined(USE_DOUBLE) || defined(USE_LDOUBLE) || defined(USE_FLOAT128)
#error "only one of USE_FLOAT, USE_DOUBLE, USE_LDOUBLE or USE_FLOAT128 can be defined"
#endif
#define FOO2 CAT2(FOO,f)
#define TYPE float
#define UTYPE uint32_t
#define EMAX 128
#define EMIN -149
#define PREC 24
#define mpfr_set_type mpfr_set_flt
#define mpfr_get_type mpfr_get_flt
#define TYPE_MAX FLT_MAX
#endif

#ifdef USE_DOUBLE
#if defined(USE_FLOAT) || defined(USE_LDOUBLE) || defined(USE_FLOAT128)
#error "only one of USE_FLOAT, USE_DOUBLE, USE_LDOUBLE or USE_FLOAT128 can be defined"
#endif
#define FOO2 FOO
#define TYPE double
#define UTYPE uint64_t
#define EMAX 1024
#define EMIN -1074
#define PREC 53
#define mpfr_set_type mpfr_set_d
#define mpfr_get_type mpfr_get_d
#define TYPE_MAX DBL_MAX
#endif

#ifdef USE_LDOUBLE
#ifdef __CUDACC__
#error "long double not supported in CUDA"
#endif
#if defined(USE_FLOAT) || defined(USE_DOUBLE) || defined(USE_FLOAT128)
#error "only one of USE_FLOAT, USE_DOUBLE, USE_LDOUBLE or USE_FLOAT128 can be defined"
#endif
#define FOO2 CAT2(FOO,l)
#define TYPE long double
#define UTYPE __uint128_t
#define EMAX 16384
#define EMIN -16445
#define PREC 64
#define mpfr_set_type mpfr_set_ld
#define mpfr_get_type mpfr_get_ld
#define TYPE_MAX LDBL_MAX
#endif

#ifdef USE_FLOAT128
#ifdef __CUDACC__
#error "binary128 not supported in CUDA"
#endif

#if defined(USE_FLOAT) || defined(USE_DOUBLE) || defined(USE_LDOUBLE)
#error "only one of USE_FLOAT, USE_DOUBLE, USE_LDOUBLE or USE_FLOAT128 can be defined"
#endif
#ifdef __INTEL_COMPILER
#define TYPE _Quad
#define FOO3 CAT2(__,FOO)
#define FOO2 CAT2(FOO3,q)
extern _Quad FOO2 (_Quad);
#define Q(x) (x ## q)
#else
#define FOO2 CAT2(FOO,f128)
#define TYPE _Float128
#define Q(x) (x ## f128)
#endif
#define TYPE_MAX Q(0xf.fffffffffffffffffffffffffff8p+16380)
#define UTYPE __uint128_t
#define EMAX 16384
#define EMIN -16494
#define PREC 113
#define mpfr_set_type mpfr_set_float128
#define mpfr_get_type mpfr_get_float128
#endif


// overload of gamma (ok in C++)
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


static void
print_type (TYPE x)
{
#ifdef USE_FLOAT
  printf ("%.8e", x);
#endif
#ifdef USE_DOUBLE
  printf ("%.16e", x);
#endif
#ifdef USE_LDOUBLE
  mpfr_t y;
  mpfr_init2 (y, PREC);
  mpfr_set_type (y, x, MPFR_RNDN);
  mpfr_printf ("%.20Re", y);
  mpfr_clear (y);
#endif
#ifdef USE_FLOAT128
  mpfr_t y;
  mpfr_init2 (y, PREC);
  mpfr_set_type (y, x, MPFR_RNDN);
  mpfr_printf ("%.35Re", y);
  mpfr_clear (y);
#endif
}

static void
print_type_hex (TYPE x)
{
#ifdef USE_FLOAT
  printf ("%a", x);
#endif
#ifdef USE_DOUBLE
  printf ("%a", x);
#endif
#ifdef USE_LDOUBLE
  mpfr_t y;
  mpfr_init2 (y, PREC);
  mpfr_set_type (y, x, MPFR_RNDN);
  mpfr_printf ("%Ral", y);
  mpfr_clear (y);
#endif
#ifdef USE_FLOAT128
  mpfr_t y;
  mpfr_init2 (y, PREC);
  mpfr_set_type (y, x, MPFR_RNDN);
  mpfr_printf ("%Ra", y);
  mpfr_clear (y);
#endif
}

typedef union { UTYPE n; TYPE x; } union_t;

inline  TYPE
get_type (UTYPE n)
{
  union_t u;
  u.n = n;
  return u.x;
}

inline UTYPE
get_utype (TYPE x)
{
  union_t u;
  u.x = x;
  return u.n;
}

static int
ndigits (UTYPE x)
{
  int n = 1; /* 0-9 have one digit */
  while (x >= 10)
    {
      x /= 10;
      n ++;
    }
  return n;
}

static void
print_utype (UTYPE x)
{
#if !defined(USE_LDOUBLE) && !defined(USE_FLOAT128)
  printf ("%lu", (unsigned long) x);
#else
  __uint128_t h, m, l;
  int first = 1;
  m = x / 10000000000UL;
  l = x % 10000000000UL;
  h = m / 10000000000UL;
  m = m % 10000000000UL;
  if (h != 0)
    {
      printf ("%lu", h);
      first = 0;
    }
  if (m != 0 || !first)
    {
      if (!first)
        {
          int n = ndigits (m);
          while (n++ < 10)
            printf ("0");
        }
      printf ("%lu", m);
    }
  if (!first)
    {
      int n = ndigits (l);
      while (n++ < 10)
        printf ("0");
    }
  printf ("%lu", l);
#endif
}

#ifdef USE_LDOUBLE
/* cf https://en.wikipedia.org/wiki/Extended_precision */
static int
is_valid (TYPE x)
{
  UTYPE n = get_utype (x);
  int e = (n >> 64) & 0x7fff; /* exponent */
  uint64_t s = (uint64_t) n;  /* significand */
  if (e == 0) return (s >> 63) == 0;
  else return (n >> 63) & 1;
}
#endif


const int maxNumOfThreads = 256;
TYPE * ypD[maxNumOfThreads];
TYPE * ypH[maxNumOfThreads];
TYPE * xpD[maxNumOfThreads];
TYPE * xpH[maxNumOfThreads];


#ifdef __CUDACC__
cudaStream_t streams[maxNumOfThreads];
__global__ void kernel_foo(TYPE const * x, TYPE * py, int bunchSize) {
   int first = blockIdx.x * blockDim.x + threadIdx.x;
   for (int i=first; i<bunchSize; i+=gridDim.x*blockDim.x) {
     py[i] = FOO2(x[i]);
   }
}
#else // CPU version
void  kernel_foo(TYPE const * x, TYPE * py, int bunchSize) {
   int first = 0;
   for (int i=first; i<bunchSize; i++) {
     py[i] = FOO2(x[i]);
   }
}
#endif

TYPE * wrap_foo(TYPE const * xH, int bunchSize) {
  int nt = omp_get_thread_num();
#ifdef __CUDACC__
  cudaCheck(cudaMemcpyAsync(xpD[nt], xH, bunchSize*sizeof(TYPE), cudaMemcpyHostToDevice, streams[nt]));
  kernel_foo<<<(bunchSize+128)/128,128,nt>>>(xpD[nt], ypD[nt],bunchSize);
  cudaCheck(cudaMemcpyAsync(ypH[nt], ypD[nt], bunchSize*sizeof(TYPE), cudaMemcpyDeviceToHost, streams[nt]));
  cudaStreamSynchronize(streams[nt]);
#else
  kernel_foo(xH, ypH[nt],bunchSize);
#endif
//  std::cout << nt << ' ' << n << ' ' << ypH[nt][0] << std::endl;
  return ypH[nt];
}


TYPE wrap_foo(TYPE x) {
  int nt = omp_get_thread_num();
  xpH[nt][0]=x;
  int bunchSize = 1;
#ifdef __CUDACC__
  cudaCheck(cudaMemcpyAsync(xpD[nt], xpH[nt], bunchSize*sizeof(TYPE), cudaMemcpyHostToDevice, streams[nt]));
  kernel_foo<<<(bunchSize+128)/128,128,nt>>>(xpD[nt], ypD[nt],bunchSize);
  cudaCheck(cudaMemcpyAsync(ypH[nt], ypD[nt], bunchSize*sizeof(TYPE), cudaMemcpyDeviceToHost, streams[nt]));
  cudaStreamSynchronize(streams[nt]);
#else
  kernel_foo(xpH[nt], ypH[nt],bunchSize);
#endif
//  std::cout << nt << ' ' << n << ' ' << ypH[nt][0] << std::endl;
  return ypH[nt][0];
}

/* return the (floating-point) distance in ulps between FOO(x) and the
   infinite precision value (estimated with MPFR and double precision) */
static void
distance (TYPE const * x, double * ulp, int bunchSize)
{
  mpfr_t xx, yy, zz;
  double ret;
  int underflow = 0;
  mpfr_exp_t expz;

#ifdef USE_LDOUBLE
  if (!is_valid (x))
    return 0;
#endif

  mpfr_set_emax (EMAX);
  /* the rounding mode of the current thread is set in doit() */
  TYPE const * z = wrap_foo(x, bunchSize);

  for (int i=0; i<bunchSize; ++i) {
    mpfr_init2 (xx, PREC);
    mpfr_init2 (yy, PREC+1);
    mpfr_init2 (zz, 2*PREC);
    mpfr_set_type (xx, x[i], MPFR_RNDN);
    MPFR_FOO (zz, xx, rnd);
    expz = mpfr_get_exp (zz);
    underflow = expz <= EMIN;
    if (isinf (z[i]) && mpfr_inf_p (zz) == 0)
    {
      /* set yy to (1-2^-(PREC+1))*2^EMAX */
      mpfr_set_ui_2exp (yy, 1, EMAX, MPFR_RNDN);
      mpfr_nextbelow (yy);
    }
    else
      mpfr_set_type (yy, z[i], MPFR_RNDN);
    mpfr_sub (zz, zz, yy, MPFR_RNDN);
    mpfr_abs (zz, zz, MPFR_RNDN);
     /* we divide by the ulp of the correctly rounded value (zz)
       which is 2^(ez-PREC) if zz is normalized, and 2^EMIN otherwise */
    if (PREC - expz <= -EMIN)
      mpfr_mul_2si (zz, zz, PREC - expz, MPFR_RNDN);
    else
      mpfr_mul_2si (zz, zz, -EMIN, MPFR_RNDN);
    ret = (underflow) ? (double) 0.0 : mpfr_get_d (zz, MPFR_RNDU);
    mpfr_clear (xx);
    mpfr_clear (yy);
    mpfr_clear (zz);
    ulp[i] = ret;
  }
}

/* return the (integer) distance in ulps between FOO(x) and the value computed
   by MPFR */
static double
ulps (TYPE x)
{
  mpfr_t xx, yy, zz;
  TYPE z;
  double ret;
  int underflow = 0;
  mpfr_exp_t expz;

#ifdef USE_LDOUBLE
  if (!is_valid (x))
    return 0;
#endif

  mpfr_set_emax (EMAX);
  /* the rounding mode of the current thread is set in doit() */

  z = wrap_foo(x);
  // z = FOO2 (x);

  mpfr_init2 (xx, PREC);
  mpfr_init2 (yy, PREC+1);
  mpfr_init2 (zz, PREC);
  mpfr_set_type (xx, x, MPFR_RNDN);
  MPFR_FOO (zz, xx, rnd);
  expz = mpfr_get_exp (zz);
  underflow = expz <= EMIN;
  if (isinf (z) && mpfr_inf_p (zz) == 0)
    {
      /* set yy to (1-2^-(PREC+1))*2^EMAX */
      mpfr_set_ui_2exp (yy, 1, EMAX, MPFR_RNDN);
      mpfr_nextbelow (yy);
    }
  else
    mpfr_set_type (yy, z, MPFR_RNDN);
  mpfr_sub (zz, zz, yy, MPFR_RNDN);
  mpfr_abs (zz, zz, MPFR_RNDN);
  /* we divide by the ulp of the correctly rounded value (zz)
     which is 2^(ez-PREC) if zz is normalized, and 2^EMIN otherwise */
  if (PREC - expz <= -EMIN)
    mpfr_mul_2si (zz, zz, PREC - expz, MPFR_RNDN);
  else
    mpfr_mul_2si (zz, zz, -EMIN, MPFR_RNDN);
  ret = (underflow) ? (double) 0.0 : mpfr_get_d (zz, MPFR_RNDU);
  mpfr_clear (xx);
  mpfr_clear (yy);
  mpfr_clear (zz);
  return ret;
}

uint64_t threshold = 1000000;
double Threshold;

/* FIXME: try lrand48_r which might be faster? */
#if RAND_MAX == 2147483647

#ifdef USE_FLOAT
static uint32_t
my_random (uint32_t n, unsigned int *seed)
{
  uint32_t ret = rand_r (seed);
  if (n > RAND_MAX)
    ret = (ret << 31) | rand_r (seed);
  return ret % n;
}
#endif
#ifdef USE_DOUBLE
static uint64_t
my_random (uint64_t n, unsigned int *seed)
{
  uint64_t ret = rand_r (seed);
  if (n > RAND_MAX)
    {
      ret = (ret << 31) | rand_r (seed);
      /* now ret <= 2^62-1 */
      if (n > 0x3fffffffffffffffLU)
        ret = (ret << 31) | rand_r (seed);
    }
  return ret % n;
}
#endif
#if defined(USE_LDOUBLE) || defined(USE_FLOAT128)
static __uint128_t
my_random (__uint128_t n, unsigned int *seed)
{
  __uint128_t ret = rand_r (seed);
  if (n > RAND_MAX)
    {
      ret = (ret << 31) | rand_r (seed);
      /* now ret <= 2^62-1 */
      if (n >> 62)
        {
          ret = (ret << 31) | rand_r (seed);
          /* now ret <= 2^93-1 */
          if (n >> 93)
            {
              ret = (ret << 31) | rand_r (seed);
              /* now ret <= 2^93-1 */
              if (n >> 124)
                ret = (ret << 31) | rand_r (seed);
            }
        }
    }
  return ret % n;
}
#endif

#else
#error "Unexpected value of RAND_MAX"
#endif

TYPE Xmax;
double Dbest = 0.0;
int Rbest = -1;
int mode_best = 0;

static void
print_error (double d)
{
  mpfr_t x;
  mpfr_init2 (x, 53);
  mpfr_set_d (x, d, MPFR_RNDU);
  if (d < 0.999)
    mpfr_printf ("%.3RUf", x);
  else if (d < 9.99)
    mpfr_printf ("%.2RUf", x);
  else if (d < 99.9)
    mpfr_printf ("%.1RUf", x);
  else
    mpfr_printf ("%.2RUe", x);
  mpfr_clear (x);
}

/* return the maximal error on [nxmin,nxmax],
   and update nxbest if it improves distance(get_type(*nxbest)) */
static double
max_heuristic1 (UTYPE nxmin, UTYPE nxmax, UTYPE *nxbest, unsigned int *seed)
{
  int nt = omp_get_thread_num();
  TYPE * x = xpH[nt];
  UTYPE nx[threshold];
  double d[threshold+1];

  TYPE xx = get_type (*nxbest);
  x[threshold] = xx;
  /* if the best value so far is in [nxmin,nxmax], use it */
  for (int i = 0; i < threshold; i++)
    {
      nx[i] = nxmin + my_random (nxmax - nxmin, seed);
      x[i] = get_type (nx[i]);
  }
  distance (x,d,threshold+1);
  double dbest = d[threshold];
  double dmax = (nxmin <= *nxbest && *nxbest < nxmax) ? dbest : 0;
  for (int i = 0; i < threshold; i++)
    {
      if (d[i] > dmax)
        {
          dmax = d[i];
          if (d[i] > dbest)
            {
              dbest = d[i];
              *nxbest = nx[i];
            }
        }
  }
  return dmax;
}

/* return the average error on [nxmin,nxmax],
   and update nxbest if it improves distance(get_type(*nxbest)) */
static double
max_heuristic2 (UTYPE nxmin, UTYPE nxmax, UTYPE *nxbest, unsigned int *seed)
{

  int nt = omp_get_thread_num();
  TYPE * x = xpH[nt];
  UTYPE nx[threshold];
  double d[threshold+1];
  double dbest;

  TYPE xx = get_type (*nxbest);
  x[threshold] = xx;
  for (int i = 0; i < threshold; i++)
    {
      nx[i] = nxmin + my_random (nxmax - nxmin, seed);
      x[i] = get_type (nx[i]);
  }
  distance (x,d,threshold);
  dbest    = d[threshold];

 double s = 0, n = 0;
  for (int i = 0; i < threshold; i++)
    {
      if (d[i] != 0)
        {
          s += d[i];
          n ++;
        }
      if (d[i] > dbest)
        {
          dbest = d[i];
          *nxbest = nx[i];
        }
    }
  if (n != 0)
    s = s / n;
  return s;
}

/* return the estimated maximal error on [nxmin,nxmax], taking into account
   mean and standard deviation,
   and update nxbest if it improves distance(get_type(*nxbest)) */
static double
max_heuristic3 (UTYPE nxmin, UTYPE nxmax, UTYPE *nxbest, unsigned int *seed)
{
  int nt = omp_get_thread_num();

  double dbest, s = 0, v = 0, n = 0;

  TYPE * x = xpH[nt];
  UTYPE nx[threshold];
  double d[threshold+1];

  TYPE xx = get_type (*nxbest);
  x[threshold] = xx;
  for (int i = 0; i < threshold; i++)
    {
      nx[i] = nxmin + my_random (nxmax - nxmin, seed);
      x[i] = get_type (nx[i]);
  }
  distance (x,d,threshold);
  dbest    = d[threshold];


  for (int i = 0; i < threshold; i++)
    {
      if (d[i] != 0)
        {
          s += d[i];
          v += d[i] * d[i];
          n ++;
        }
      if (d[i] > dbest)
        {
          dbest = d[i];
          *nxbest = nx[i];
        }
    }
  
  /* compute mean and standard deviation */
  if (n != 0)
    {
      s = s / n;
      v = v / n - s * s;
      if (v < 0)
        v = 0;
      double sigma = sqrt (v);
      /* we got n non-zero values out of threshold, thus we should get
         n/threshold*(nxmax-nxmin) */
      n = n * (double) (nxmax - nxmin) / (double) threshold;
      double logn = log (n);
      double t = sqrt (2.0 * logn);
      /* Reference: A note on the first moment of extreme order statistics
         from the normal distribution, Max Petzold,
         https://gupea.ub.gu.se/handle/2077/3092 */
      s = s + sigma * (t - (log (logn) + 1.3766) / (2.0 * t));
    }
  return s;
}

#ifdef NO_OPENMP
static int
omp_get_num_threads (void)
{
  return 1;
}

static int
omp_get_thread_num (void)
{
  return 0;
}
#endif

static int
mode (void)
{
  if (use_mode != -1)
    return use_mode;
  int i = omp_get_thread_num ();
  /* Modes 1 and 2 seem to be less efficient (for example for acosh with
     threshold=10000 they give only 0.983603 whereas mode 0 gives 2.18-2.19),
     thus we use only one thread on each. */
  if (i == 1 || i == 2)
    return i;
  return 0;
}

#ifdef STAT
unsigned long eval_heuristic = 0;
unsigned long eval_exhaustive = 0;
#endif

static double
max_heuristic (UTYPE nxmin, UTYPE nxmax, UTYPE *nxbest, unsigned int *seed)
{
  int k = mode ();

#ifdef STAT
#pragma omp atomic update
  eval_heuristic += threshold;
#endif

  if (k == 0)
    return max_heuristic1 (nxmin, nxmax, nxbest, seed);
  else if (k == 1)
    return max_heuristic2 (nxmin, nxmax, nxbest, seed);
  else /* k = 2 */
    return max_heuristic3 (nxmin, nxmax, nxbest, seed);
}

typedef struct {
  UTYPE nxmin, nxmax;
  double d;
#ifdef RANK
  int rank; /* worst rank */
#endif
} chunk_t;

static double
chunk_size (chunk_t c)
{
  return (double) (c.nxmax - c.nxmin);
}

static void
chunk_swap (chunk_t *a, chunk_t *b)
{
  UTYPE tmp;
  tmp = a->nxmin; a->nxmin = b->nxmin; b->nxmin = tmp;
  tmp = a->nxmax; a->nxmax = b->nxmax; b->nxmax = tmp;
  double ump;
  ump = a->d; a->d = b->d; b->d = ump;
#ifdef RANK
  int rmp;
  rmp = a->rank; a->rank = b->rank; b->rank = rmp;
#endif
}

typedef struct {
  chunk_t *l;
  int size;
} List_struct;
typedef List_struct List_t[1];

/* Idea from Eric Schneider: instead of sampling only one interval at each
   level of the binary splitting tree, we sample up to LIST_ALLOC intervals,
   and keep the most promising ones. We use the default value suggested by
   Eric Schneider (20). */
#define LIST_ALLOC 20

static void
List_init (List_t L)
{
  L->l = (chunk_t*)malloc (LIST_ALLOC * sizeof (chunk_t));
  L->size = 0;
}

static void
List_init2 (List_t L, UTYPE nxmin, UTYPE nxmax)
{
  L->l = (chunk_t*)malloc (LIST_ALLOC * sizeof (chunk_t));
  L->l[0].nxmin = nxmin;
  L->l[0].nxmax = nxmax;
  L->size = 1;
}

static void
List_print (List_t L)
{
  int i;
  for (i = 0; i < L->size; i++)
    printf ("%.3g ", L->l[i].d);
  printf ("\n");
}

static void
List_check (List_t L)
{
  int i;
  for (i = 1; i < L->size; i++)
    if (L->l[i-1].d < L->l[i].d)
      {
        fprintf (stderr, "Error, list not sorted:\n");
        List_print (L);
        exit (1);
      }
}

static void
List_insert (List_t L, UTYPE nxmin, UTYPE nxmax, double d)
{
  int i = L->size;
  if (i < LIST_ALLOC || d > L->l[LIST_ALLOC-1].d)
    {
      /* if list is not full, we insert at position i,
         otherwise we insert at position LIST_ALLOC-1 */
      if (i == LIST_ALLOC) /* replaces previous chunk */
        i--;
      else
        L->size++;
      L->l[i].nxmin = nxmin;
      L->l[i].nxmax = nxmax;
      L->l[i].d = d;
      /* now insertion sort */
      while (i > 0 && d > L->l[i-1].d)
        {
          chunk_swap (L->l + (i-1), L->l + i);
#ifdef RANK
          if (L->l[i].rank < i)
            L->l[i].rank = i; /* updates maximal rank */
#endif
          i--;
        }
#ifdef RANK
      L->l[i].rank = i; /* set initial rank */
#endif
    }
  // List_check (L);
}

static void
List_swap (List_t L1, List_t L2)
{
  chunk_t *tmp;
  tmp = L1->l; L1->l = L2->l; L2->l = tmp;
  int ump;
  ump = L1->size; L1->size = L2->size; L2->size = ump;
}

static void
List_clear (List_t L)
{
  free (L->l);
}

static void
exhaustive_search (chunk_t *c, UTYPE *nxbest, double *dbest, int *rbest)
{

  int bunchSize = c->nxmax - c->nxmin;

  int nt = omp_get_thread_num();

  // printf("exha %d %d\n",nt,bunchSize);

  TYPE * x = xpH[nt];
  UTYPE nx[bunchSize];
  double d[bunchSize+1];

  TYPE xx = get_type (*nxbest);
  x[bunchSize] = xx;
  auto nxmin =  c->nxmin;
  for (int i = 0; i < bunchSize; i++)
    {
      nx[i] = nxmin + i;
      x[i] = get_type (nx[i]);
  }
  distance (x,d,bunchSize+1);
  (*dbest) = d[bunchSize];

  for (int i = 0; i < bunchSize; i++)
    {
      if (d[i] > *dbest)
        {
          *dbest = d[i];
          *nxbest = nx[i];
#ifdef RANK
          *rbest = c->rank;
#endif
        }
    }
#ifdef STAT
#pragma omp atomic update
  eval_exhaustive += c->nxmax - c->nxmin;
#endif
}

/* search for nxmin <= nx < nxmax
   where the worst found so far is (nxbest,dbest) */
static void
search (UTYPE nxmin, UTYPE nxmax, UTYPE *nxbest, double *dbest, int *rbest,
        unsigned int *seed)
{
  List_t L;

  List_init2 (L, nxmin, nxmax);
  while (1)
    {
      assert (1 <= L->size && L->size <= LIST_ALLOC);
      double width = chunk_size (L->l[0]);
      if (width <= Threshold) /* exhaustive search */
        {
          int i;
          for (i = 0; i < L->size; i++)
            exhaustive_search (L->l + i, nxbest, dbest, rbest);
          break;
        }
      else /* split each chunk in two */
        {
          List_t NewL;
          int i;
          List_init (NewL);
          for (i = 0; i < L->size; i++)
            {
              UTYPE nxmin = L->l[i].nxmin;
              UTYPE nxmax = L->l[i].nxmax;
              UTYPE nxmid = nxmin + (nxmax - nxmin) / 2;
              double d1 = max_heuristic (nxmin, nxmid, nxbest, seed);
              List_insert (NewL, nxmin, nxmid, d1);
              double d2 = max_heuristic (nxmid, nxmax, nxbest, seed);
              List_insert (NewL, nxmid, nxmax, d2);
            }
          List_swap (L, NewL);
#ifdef STAT
          printf ("L: "); List_print (L);
#endif
          List_clear (NewL);
        }
    }
  List_clear (L);
}

static void
doit (unsigned int seed)
{
  UTYPE nxbest = 0;
  double dbest = 0;
  int rbest = -1;

  /* set the rounding mode of the current thread */
  fesetround (rnd1[rnd]);

  /* for thread 0, get the "worst" value so far as initial point */
  if (omp_get_thread_num () == 0)
    {
      dbest = Dbest;
      nxbest = get_utype (Xmax);
    }

  /* assume -xxx_MAX has the largest encoding */
  UTYPE bound = get_utype (-TYPE_MAX) + 1;
  search (0, bound, &nxbest, &dbest, &rbest, &seed);
#pragma omp critical
  if (dbest > Dbest)
    {
      Dbest = dbest;
#ifdef RANK
      Rbest = rbest;
#endif
      mode_best = mode ();
      Xmax = get_type (nxbest);
    }
  mpfr_free_cache (); /* free cache of current thread */
}

static void
init_Threshold (void)
{
  UTYPE bound = get_utype (-TYPE_MAX) + 1;
  double w = (double) bound; /* width of current interval */
  double s = 0;              /* number of evaluations so far */
  while (s < w)
    {
      w = w / 2.0;
      s += 2 * (double) threshold;
    }
  Threshold = w;
}

int
main (int argc, char *argv[])
{
  int verbose = 0;
  long seed = 0;

  while (argc >= 2 && argv[1][0] == '-')
    {
      if (argc >= 3 && strcmp (argv[1], "-threshold") == 0)
        {
          threshold = strtoul (argv[2], NULL, 10);
          argv += 2;
          argc -= 2;
        }
      else if (argc >= 3 && strcmp (argv[1], "-seed") == 0)
        {
          seed = strtoul (argv[2], NULL, 10);
          argv += 2;
          argc -= 2;
        }
      else if (argc >= 3 && strcmp (argv[1], "-mode") == 0)
        {
          use_mode = strtoul (argv[2], NULL, 10);
          assert (0 <= use_mode && use_mode <= 2);
          argv += 2;
          argc -= 2;
        }
      else if (strcmp (argv[1], "-v") == 0)
        {
          verbose = 1;
          argv ++;
          argc --;
        }
      else if (strcmp (argv[1], "-rndn") == 0)
        {
          rnd = mpfr_rnd_t(0);
          argc --;
          argv ++;
        }
      else if (strcmp (argv[1], "-rndz") == 0)
        {
          rnd = mpfr_rnd_t(1);
          argc --;
          argv ++;
        }
      else if (strcmp (argv[1], "-rndu") == 0)
        {
          rnd = mpfr_rnd_t(2);
          argc --;
          argv ++;
        }
      else if (strcmp (argv[1], "-rndd") == 0)
        {
          rnd = mpfr_rnd_t(3);
          argc --;
          argv ++;
        }
      else
        {
          fprintf (stderr, "Unknown option %s\n", argv[1]);
          exit (1);
        }
    }
  assert (threshold > 0);
  /* divide threshold by LIST_ALLOC so that the total number of evalutions
     does not vary with LIST_ALLOC */
  threshold = 1 + (threshold - 1) / LIST_ALLOC;
  init_Threshold ();

  int bunchSize = (threshold >int(Threshold+1)) ? threshold : int(Threshold+1);
  int nstreams = omp_get_max_threads();
  assert(maxNumOfThreads>nstreams);
 if (verbose)
   printf("thresholds %d %f\n",threshold,Threshold);
   
#ifdef __CUDACC__
  for (int i = 0; i < nstreams; i++)
    {
        cudaCheck(cudaStreamCreate(&(streams[i])));
        cudaCheck(cudaMalloc((void **)&ypD[i], bunchSize*sizeof(TYPE)));
        cudaCheck(cudaMallocHost((void **)&ypH[i], bunchSize*sizeof(TYPE)));
        cudaCheck(cudaMalloc((void **)&xpD[i], bunchSize*sizeof(TYPE)));
        cudaCheck(cudaMallocHost((void **)&xpH[i], bunchSize*sizeof(TYPE)));

    }
#else
 for (int i = 0; i < nstreams; i++)
    {
      ypH[i] = (TYPE *)malloc(bunchSize*sizeof(TYPE));
      xpH[i] = (TYPE *)malloc(bunchSize*sizeof(TYPE));
    }
#endif

    {
      int nt = omp_get_thread_num();
      TYPE x = 0.5;
      TYPE y = wrap_foo(x);
      // y = FOO2 (Xmax);
      printf ("\n%d at %a ", nt, x);
      printf ("libm gives %a\n\n", y);
    }


 if (verbose)
    {
#ifdef GLIBC
  if (verbose)
      printf("GNU libc version: %s\n", gnu_get_libc_version ());
      printf("GNU libc release: %s\n", gnu_get_libc_release ());
#elif __CUDACC__
#ifndef CUDART_VERSION
 #warning "no " CUDART_VERSION
#endif
    printf ("Using CUDA %d\n",CUDART_VERSION);
    int cuda_device = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, cuda_device);
    printf("CUDA Capable: SM %d.%d hardware\n", deviceProp.major, deviceProp.minor);
#endif
    printf ("MPFR library: %-12s\nMPFR header:  %s (based on %d.%d.%d)\n",
            mpfr_get_version (), MPFR_VERSION_STRING, MPFR_VERSION_MAJOR,
            MPFR_VERSION_MINOR, MPFR_VERSION_PATCHLEVEL);
    printf ("Checking function %s with %s\n", NAME,
            mpfr_print_rnd_mode (rnd2[rnd]));
  fflush (stdout);
  }  // verbose

  if (seed == 0)
    seed = getpid ();
  if (verbose)
    printf ("Using seed %lu\n", seed);

#if defined(USE_FLOAT) || defined(USE_DOUBLE) || defined(USE_LDOUBLE)
#define NLIBS 8
#else
#define	NLIBS 2 /* only glibc and icc do provide binary128 */
#endif
#define SIZE (30*NLIBS)

#ifdef WORST
#ifdef GLIBC
#define NUMBER 0
#endif
#ifdef ICC
#define NUMBER 1
#endif
#ifdef AMD
#define NUMBER 2
#endif
#ifdef NEWLIB
#define NUMBER 3
#endif
#ifdef OPENLIBM
#define NUMBER 4
#endif
#ifdef MUSL
#define NUMBER 5
#endif
#ifdef APPLE
#define NUMBER 6
#endif
#ifdef  __CUDACC__
#define NUMBER 7
#endif
#ifndef NUMBER
#warning  "Library undefined. Set to GLIBC"
#define GLIBC
#define NUMBER 0
#endif

#define SIZE_EXTRA 12
#ifdef USE_FLOAT
  TYPE worst[SIZE] = {
  };
TYPE extra[SIZE_EXTRA] = {
  0
  };
#endif
#ifdef USE_DOUBLE
  TYPE worst[SIZE] = {
    /* acos */
#ifndef GLIBC_NEW
    0x1.fffff3634acd6p-1,   /* GNU libc 2.32 0.5000000044534775 */
#else /* patchset 41668 from Jan 7, 2021 */
    0x1.dffff776c7505p-1,   /* 0.5221736284948646 */
#endif
    0x1.6c05eb219ec46p-1,   /* icc 19.1.3.304 */
    -0x1.01c3d975759c3p-1,  /* AMD LibM */
    -0x1.004118e0783e9p-1,  /* Newlib */
    -0x1.002ed02c0c632p-1,  /* OpenLibm 0.7.0 */
    -0x1.002ed02c0c632p-1,  /* Musl */
    -0x1.8d313198a2e02p-53, /* Apple 1.05153 */
    0x1.26665e8b2d487p-1,  /* CUDA 11.2 */
    /* acosh */
    0x1.0001ff6afc4bap+0,   /* GNU libc */
    0x1.01825ca7da7e5p+0,   /* icc 19.1.3.304 */
    0x1.40c044a37af12p+0,   /* AMD LibM */
    0x1.0001ff6afc4bap+0,   /* Newlib */
    0x1.0001ff6afc4bap+0,   /* OpenLibm 0.7.0 */
    0x1.0001ff6afc4bap+0,   /* Musl 1.2.1 */
    0x1.001f1234c557p+0,    /* Apple 2.19887 */ 
    0x1.1e2b17a0868aap+0,  /* CUDA 11.2 */
    /* asin */
#ifndef GLIBC_NEW
    0x1.f6e5a0d80501ap-4,   /* GNU libc 0.500017 */
#else
    0x1.07fffeadab7ecp-3,   /* 0.5154024135462042 */
#endif
    0x1.6c042a6378102p-1,   /* icc 19.1.3.304 */
    -0x1.0181c4fd1ff3fp-1,  /* AMD LibM */
    -0x1.004d1c5a9400bp-1,  /* Newlib */
    -0x1.004d1c5a9400bp-1,  /* OpenLibm 0.7.0 */
    -0x1.004d1c5a9400bp-1,  /* Musl 1.2.1 */
    -0x1.eae73c700c2f6p-2,  /* Apple 0.743983 */
     0x1.2ef63ec3ba1fap-1,  /* CUDA 11.2 */
    /* asinh */
    -0x1.02657ff36d5f3p-2,  /* GNU libc */
    -0x1.00064c7d4d16ep-4,  /* icc */
    -0x1.00238008c0c6fp+0,  /* AMD LibM */
    -0x1.02657ff36d5f3p-2,  /* Newlib */
    -0x1.02657ff36d5f3p-2,  /* OpenLibm 0.7.0 */
    -0x1.0240f2bdb3f25p-2,  /* Musl 1.2.1 */
    0x1.fadea351d1db1p-3,   /* Apple 1.56419 */
    -0x1.091f7703d0773p-1,  /* CUDA 11.2 */
    /* atan */
#ifndef GLIBC_NEW
    0x1.20538781986c3p+51,  /* GNU libc 0.5 */
#else
    -0x1.f9004176e07bep-4,  /* 0.5221662545321709 */
#endif
    -0x1.ffff8020d3d1dp-7,  /* icc 19.1.3.304 */
    0x1.601c5ffc19b43p-1,   /* AMD LibM */
    0x1.62ff6a1682c25p-1,   /* Newlib */
    -0x1.607f02c43788dp-1,  /* OpenLibm */
    -0x1.607f02c43788dp-1,  /* Musl */
    0x1.13958c08ab64ap+1,   /* Apple 0.863559 */
    -0x1.54cde4bc5a521p+0,  /* CUDA 11.2 */
   /* atanh */
    -0x1.f97fabc0650c4p-4,  /* GNU libc */
    -0x1.e93525563e6ffp-9,  /* icc */
    -0x1.34144d5b303b4p-1,  /* AMD LibM */
    -0x1.f97fabc0650c4p-4,  /* Newlib */
    -0x1.eb21a5af3f144p-4,  /* OpenLibm 0.7.0 */
    -0x1.f8a404597baf4p-4,  /* Musl 1.2.1 */
    0x1.ffd834a270fp-10,    /* Apple 2.00021 */
    0x1.f55ef33b32a1ep-3 ,  /* CUDA 11.2 */
   /* cbrt */
    0x1.7a13d2b82be1ap-254,  /* GNU libc */
    0x1.f7b65d51efca8p+548,  /* icc 19.1.3.304 */
    0x1.09806cdccbfa1p-748,  /* AMD LibM */
    0x1.011c793d2b4fap+681,  /* Newlib */
    -0x1.2bf9d2510bed4p+798, /* OpenLibm 0.7.0 */
    -0x1.2bf9d2510bed4p+798, /* Musl 1.2.1 */
    0x1.facda6d71fdcfp-988,  /* Apple 0.728607 */
    0x1.8bea12d76753p+834,  /* CUDA 11.2 */
    /* cos */
    0x1.1feeccb9e4bf7p+5,    /* GNU libc */
    -0x1.1f766f67555c1p+104, /* icc 19.1.3.304 */
    0x1.293f72677a7e7p+13,   /* AMD LibM */
    -0x1.4ae182c1ab422p+21,  /* Newlib */
    -0x1.34e729fd08086p+21,  /* OpenLibm */
    -0x1.34e729fd08086p+21,  /* Musl */
    0x1.2f29ff7e1c49fp+7,    /* Apple 0.945054 */
    0x1.8ad56f78d6f4ap+6,  /* CUDA 11.2 */
    /* cosh */
    -0x1.633c654fee2bap+9,   /* GNU libc */
    -0x1.5a364e6b98134p+9,   /* icc 19.1.3.304 */
    0x1.1ffff6b05a77fp+4,    /* AMD LibM */
    0x1.633cc2ae1c934p+9,    /* Newlib */
    -0x1.6310ab92794a8p+9,   /* OpenLibm 0.7.0 */
    0x1.502b6c6156f9fp+0,    /* Musl 1.2.1 */
    -0x1.62c79cd1d386cp-2,   /* Apple 0.522110 */
    0x1.e7fb557543454p+1,  /* CUDA 11.2 */
    /* erf */
    0x1.c332bde7ca515p-5,    /* GNU libc */
    0x1.00b4ce467e371p+2,    /* icc */
    0x1.c332bde7ca515p-5,    /* AMD LibM */
    -0x1.c57541b55c8ebp-16,  /* Newlib */
    -0x1.c57541b55c8ebp-16,  /* OpenLibm 0.7.0 */
    -0x1.c57541b55c8ebp-16,  /* Musl 1.2.1 */
    -0x1.e1500cfaad20ep-2,   /* Apple 6.16980 */
    0x1.321a11445f63fp-2,  /* CUDA 11.2 */
    /* erfc */
    0x1.3ff86013cf2bap+0,    /* GNU libc */
    0x1.5d7f2dfe3b94ep-1,    /* icc 19.1.3.304 */
    0x1.3ff86013cf2bap+0,    /* AMD LibM */
    0x1.531fe30327333p+0,    /* Newlib */
    0x1.531fe30327333p+0,    /* OpenLibm 0.7.0 */
    0x1.527f4fb0d9331p+0,    /* Musl 1.2.1 */
    0x1.bba28df12b2d7p+1,    /* Apple 10.3248 */
    0x1.f666ebe136c4ap-7,  /* CUDA 11.2 */
    /* exp */
    -0x1.571eb1496dab3p+9,   /* GNU libc */
    0x1.fce66609f7428p+5,    /* icc 19.1.3.304 */
    -0x1.6237c669d1a9fp+9,   /* AMD LibM */
    0x1.6f5ea0e012c38p+6,    /* Newlib */
    0x1.6f5ea0e012c38p+6,    /* OpenLibm 0.7.0 */
    -0x1.1820ac5084912p+6,   /* Musl 1.2.1 */
    -0x1.f8ee4b949bf25p-29,  /* Apple 0.519719 */
    0x1.e7fb382169bc1p+1,  /* CUDA 11.2 */
    /* exp10 */
    0x1.334ab33a9aaep-2,     /* GNU libc */
    0x1.302160f9d4c47p+0,    /* icc */
    -0x1.33b58776304ebp+8,   /* AMD LibM */
    0x1.3450246086766p-3,    /* Newlib */
    0,                       /* OpenLibm 0.7.0 */
    -0x1.fe8cc6dccb491p+3,   /* Musl 1.2.1 */
    0,                       /* Apple */
    0x1.5c1ef9dc31a01p+0,  /* CUDA 11.2 */
    /* exp2 */
    -0x1.1b214e75178c4p-5,   /* GNU libc */
    0x1.f3ffd85f33423p-1,    /* icc 19.1.3.304 */
    -0x1.ff03ffe8ed867p+9,   /* AMD LibM */
    0x1.003d313b3426fp-1,    /* Newlib */
    -0x1.ff1eb5acee46bp+9,   /* OpenLibm 0.7.0 */
    -0x1.1b214e75178c4p-5,   /* Musl */
    -0x1.d2bfcb0ff982fp-13,  /* Apple 0.519617 */
    0x1.ff3d7b8f8198p+9,  /* CUDA 11.2 */
    /* expm1 */
    0x1.62f69d171fa65p-2,    /* GNU libc */
    -0x1.635e445cc416bp-8,   /* icc 19.1.3.304 */
    0x1.9a579b4e7005fp-2,    /* AMD LibM */
    0x1.63daa4e2b9346p-2,    /* Newlib */
    0x1.63d05581e2f84p-2,    /* OpenLibm 0.7.0 */
    0x1.63d05581e2f84p-2,    /* Musl 1.2.2 */
    0x1.e7ebb59408602p-5,    /* Apple 0.702290 */
    0x1.2984a10a3c349p+5,  /* CUDA 11.2 */
    /* j0 */
    0x1.33d152e971b4p+1,     /* GNU libc */
    0x1.d6d8a4b5aa0e2p+9,    /* icc */
    0x1.33d152e971b4p+1,     /* AMD LibM */
    -0x1.45f306d16c7cap+915, /* Newlib */
    0x1.33d152e971b4p+1,     /* OpenLibm 0.7.0 */
    -0x1.33d152e971b4p+1,    /* Musl 1.2.1 */
    0x1.33d152e971b4p+1,     /* Apple */
   -0x1.03163dca16356p+25,  /* CUDA 11.2 */
    /* j1 */
    -0x1.ea75575af6f09p+1,   /* GNU libc */
    -0x1.67b5541c7d8b7p+7,   /* icc 19.1.3.304 */
    -0x1.ea75575af6f09p+1,   /* AMD LibM */
    -0x1.45f306dc1656cp+759, /* Newlib */
    -0x1.ea75575af6f09p+1,   /* OpenLibm 0.7.0 */
    0x1.ea75575af6f09p+1,    /* Musl 1.2.1 */
    -0x1.ea75575af6f09p+1,   /* Apple */
    -0x1.6b6d192ad2eefp+26,  /* CUDA 11.2 */
    /* lgamma */
    -0x1.f60a8d4969457p+1,   /* GNU libc */
    -0x1.3f62c60e23b31p+2,   /* icc 19.1.3.304 */
    -0x1.f60a8d4969457p+1,   /* AMD LibM */
    -0x1.3a7fc9600f86cp+1,   /* RedHat Newlib 4.0.0 */
    -0x1.3a7fc9600f86cp+1,   /* OpenLibm 0.7.0 */
    -0x1.3a7fc9600f86cp+1,   /* Musl 1.2.1 */
    -0x1.bffcbf76b86fp+2,    /* Apple 2.32471e+16 */
    -0x1.fa471547c2fe5p+1,  /* CUDA 11.2 */
    /* log */
    0x1.1211bef8f68e9p+0,    /* GNU libc */
    0x1.008000db2e8bep+0,    /* icc 19.1.3.304 */
    0x1.0ffda7808ae17p+0,    /* AMD LibM */
    0x1.48a807fa56469p+0,    /* Newlib */
    0x1.48652aa7cf249p+0,    /* OpenLibm 0.7.0 */
    0x1.dc06b15bb66e2p-1,    /* Musl */
    0x1.43096728988p-1,      /* Apple 0.507836 */
    0x1.689ef337a0589p-1,  /* CUDA 11.2 */
    /* log10 */
    0x1.de00583c54794p-1,    /* GNU libc */
    0x1.feda7b62c1033p-1,    /* icc 19.1.3.304 */
    0x1.e0030380b742fp-1,    /* AMD LibM */
    0x1.553eb6579b261p+0,    /* Newlib */
    0x1.5536e3c4fc059p+0,    /* OpenLibm 0.7.0 */
    0x1.5536e3c4fc059p+0,    /* Musl 1.2.1 */
    0x1.25025b738b51dp-1,    /* Apple 0.512936 */
    0x1.8c223f0a82905p+421,  /* CUDA 11.2 */
    /* log1p */
    -0x1.2c10396268852p-2,   /* GNU libc */
    0x1.000aee2a2757fp-9,    /* icc */
    0x1.10734fffffff9p-4,    /* AMD LibM */
    -0x1.2bf5e2b0dab12p-2,   /* Newlib */
    -0x1.2bf5e2b0dab12p-2,   /* OpenLibm */
    -0x1.2c19ff3584c92p-2,   /* Musl */
    -0x1.fffffb3ffffa5p-28,  /* Apple 0.666667 */
    -0x1.fffffa9bab3e5p-2,  /* CUDA 11.2 */
    /* log2 */
    0x1.0b534e1fc0fe2p+0,    /* GNU libc */
    0x1.00ffdc158ff0dp+0,    /* icc 19.1.3.304 */
    0x1.e007c3c7bee53p-1,    /* AMD LibM */
    0x1.68a75af163d9dp+0,    /* Newlib */
    0x1.69ca14ce191b7p+0,    /* OpenLibm 0.7.0 */
    0x1.0b548b52d2c46p+0,    /* Musl 1.2.1 */
    0x1.6b01a081c3022p-1,    /* Apple 0.514464 */
    0x1.40a116c951bp-215,  /* CUDA 11.2 */
    /* sin */
    -0x1.f8b791cafcdefp+4,   /* GNU libc */
    -0x1.0e16eb809a35dp+944, /* icc 19.1.3.304 */
    -0x1.f05e952d81b89p+5,   /* AMD LibM */
    -0x1.842d8ec8f752fp+21,  /* Newlib */
    0x1.b8b6d07237443p+21,   /* OpenLibm 0.7.0 */
    0x1.b8b6d07237443p+21,   /* Musl 1.2.1 */
    -0x1.07e432e88a6e7p+4,   /* Apple 0.940497 */
    0x1.cfa11ebe115dcp+16,  /* CUDA 11.2 */
    /* sinh */
    -0x1.633c654fee2bap+9,   /* GNU libc */
    -0x1.adc135eb544c1p-2,   /* icc */
    0x1.1ff76b878859ap+3,    /* AMD LibM */
    -0x1.633cae1335f26p+9,   /* Newlib */
    0x1.6320943636f24p-1,    /* OpenLibm 0.7.0 */
    0x1.6320943636f24p-1,    /* Musl 1.2.1 */
    0x1.623853046e994p-2,    /* Apple 0.534360 */
    0x1.bda830d0e2c74p+0,  /* CUDA 11.2 */
    /* sqrt */
    0x1.fffffffffffffp-1,    /* GNU libc */
    0x1.fffffffffffffp-1,    /* icc 19.1.3.304 */
    0x1.fffffffffffffp-1,    /* AMD LibM */
    0x1.fffffffffffffp-1,    /* Newlib */
    0x1.fffffffffffffp-1,    /* OpenLibm 0.7.0 */
    0x1.fffffffffffffp-1,    /* Musl 1.2.1 */
    0x1.fffffffffffffp-1,    /* Apple */
    0x1.e91c9d18b0096p+353,  /* CUDA 11.2 */
    /* tan */
#ifndef GLIBC_NEW
    0x1.50486b2f87014p-5,    /* GNU libc 2.32 0.5 */
#else
    0x1.c673a473b3503p+3,    /* 0.6183652837454632 */
#endif
    0x1.4d314589ddb04p+18,   /* icc 19.1.3.304 */
    -0x1.6842486cdd221p+12,  /* AMD LibM */
    0x1.3f9605aaeb51bp+21,   /* Newlib */
    0x1.3f9605aaeb51bp+21,   /* OpenLibm 0.7.0 */
    0x1.3f9605aaeb51bp+21,   /* Musl 1.2.2 */
    -0x1.a81d88f3375ep+6,    /* Apple 3.49177 */
    -0x1.fd08517eeabf9p+22,  /* CUDA 11.2 */
    /* tanh */
    0x1.e0f65b6ce8826p-3,    /* GNU libc */
    -0x1.0018308fc500dp+0,   /* icc */
    -0x1.ff58c8cce5385p-1,   /* AMD LibM */
    0x1.e0f65b6ce8826p-3,    /* Newlib */
    0x1.e100f835705efp-3,    /* OpenLibm 0.7.0 */
    0x1.e100f835705efp-3,    /* Musl 1.2.1 */
    0x1.00cf9f273d84p+1,     /* Apple 0.612229 */
    0x1.1917834b6f97bp-1,  /* CUDA 11.2 */
    /* tgamma */
    -0x1.c033cc426752fp+2,   /* GNU libc */
    -0x1.3e002bee87875p+6,   /* icc 19.1.3.304 */
    -0x1.c033cc426752fp+2,   /* AMD LibM */
    -0x1.53dc3682f7e9ap+7,   /* Newlib */
    -0x1.540b170c4e65ep+7,   /* OpenLibm 0.7.0 */
    -0x1.ff29534000245p+2,   /* Musl */
    -0x1.55f5b67b0b1edp+7,   /* Apple */
    -0x1.2baa15eec9f9bp+7,  /* CUDA 11.2 */
    /* y0 */
    0x1.c982eb8d417eap-1,    /* GNU libc */
    0x1.f78ea64a68b96p-63,   /* icc */
    0x1.c982eb8d417eap-1,    /* AMD LibM */
    0x1.c982eb8d417eap-1,    /* Newlib */
    0x1.c982eb8d417eap-1,    /* OpenLibm 0.7.0 */
    0x1.c982eb8d417eap-1,    /* Musl 1.2.1 */
    0x1.c982eb8d417eap-1,    /* Apple */
    0x1.337dfc96af776p+25,  /* CUDA 11.2 */
    /* y1 */
    0x1.193bed4dff243p+1,    /* GNU libc */
    0x1.c50658fc9bc2dp+0,    /* icc */
    0x1.193bed4dff243p+1,    /* AMD LibM */
    0x1.193bed4dff243p+1,    /* Newlib */
    0x1.193bed4dff243p+1,    /* OpenLibm 0.7.0 */
    0x1.193bed4dff243p+1,    /* Musl 1.2.1 */
    0x1.193bed4dff243p+1,    /* Apple */
    0x1.2333038fbcb7ep+26,  /* CUDA 11.2 */
  };
TYPE extra[SIZE_EXTRA] = {
  /* don't remove the following values: they should give an error > 0.5 for
     glibc asin after commit f67f9c9 */
  0x1.fcd5742999ab8p-1,
  -0x1.ee2b43286db75p-1,
  -0x1.f692ba202abcp-4,
  -0x1.9915e876fc062p-1,
  -0x1.fd7d13f1663afp-1, /* 5.0000005122065272e-01 for asin after f67f9c9 */
  0x1.16c08b622e36p-1, /* 4.9999999999919309e-01 for asin with glibc-2.32 */
  /* same for glibc acos */
  0x1.f63845056f35ep-1,
  0x1.fffff3634acd6p-1, /* glibc-2.32 acos 0.5000000044534775 */
  -0x1.8814da6eb7dbp+5,    /* 8.78636 [9] with rndn and branch tgamma */
  -0x1.8814da6eb7dbp+5,    /* 8.78636 [9] with rndz and branch tgamma */
  -0x1.8814da6eb7dbp+5,    /* 8.78636 [8] with rndu and branch tgamma */
  -0x1.8814da6eb7dbp+5,    /* 8.78636 [9] with rndd and branch tgamma */
  };
#endif
#ifdef USE_LDOUBLE
/* AMD Libm does not provide long double functions */
  TYPE worst[SIZE] = {
    /* acos */
    0xf.fe00271d507ee5dp-4l,   /* glibc 1.74154 */
    0x8.af256cd27462348p-4l,   /* icc 0.504696 */
    0,                         /* AMD */
    0,                         /* Newlib */
    -0x8.0554265fff341fp-4l,   /* OpenLibm 0.935979 */
    0xf.fe00271d507ee5dp-4l,   /* Musl 1.74154 */
    0,                         /* Apple */
    /* acosh */
    0x1.1ecb0b4b2ea31386p+0l,  /* glibc 2.97937 */
    0x1.1f9c4feedfe4f2cp+0l,   /* icc 0.501721 */
    0,                         /* AMD */
    0,                         /* Newlib */
    0x1.103210d403802b08p+0l,  /* OpenLibm 3.10964 */
    0x1.1ecb0b4b2ea31386p+0l,  /* Musl 2.97937 */
    0,                         /* Apple */
    /* asin */
    0x8.171fd358c4cb27bp-4l,   /* glibc 1.1424 */
    -0x8.136ad28a7021398p-4l,  /* icc 0.505447 */
    0,                         /* AMD */
    0,                         /* Newlib */
    0x8.0519515d1e15a6bp-4l,   /* OpenLibm 1.02829 */
    -0x3.fff0a397b8dea17cp-8l, /* Musl 1.99565 */
    0,                         /* Apple */
    /* asinh */
    -0x8.0bb656992eac437p-4l,  /* glibc 2.95329 */
    -0x7.ff026e0791094718p-4l, /* icc 0.505662 */
    0,                         /* AMD */
    0,                         /* Newlib */
    -0x5.c9866cb231f2c7c8p-4l, /* OpenLibm 3.18771 */
    0x8.17d42c799f71abfp-4l,   /* Musl 2.95261 */
    0,                         /* Apple */
    /* atan */
    -0x1.0411ae010d4c5b1ep+0l, /* glibc 0.639295 */
    -0x8.00f60592e42d79p+8l,   /* icc 0.500347 */
    0,                         /* AMD */
    0,                         /* Newlib */
    0x6.fffee696b91270fp-4l,   /* OpenLibm 1.09611 */
    -0x1.0411ae010d4c5b1ep+0l, /* Musl 0.639295 */
    0,                         /* Apple */
    /* atanh */
    -0x3.34344daa8d4b4038p-4l, /* glibc 2.86941 */
    0x3.e7bdca850921266p-4l,   /* icc 0.500522 */
    0,                         /* AMD */
    0,                         /* Newlib */
    -0xf.ffffffffffffe78p-32l, /* OpenLibm 85.3333 */
    -0x3.339d9fda8adc2d78p-4l, /* Musl 3.18775 */
    0,                         /* Apple */
    /* cbrt */
    0xc.f5414f1e9486ffep+16084l, /* glibc 0.823365 */
    -0x2.320375fd33ed311cp-13376l,/* icc 0.502427 */
    0,                           /* AMD */
    0,                           /* Newlib */
    -0x3.fffffffa5623708p+4588l, /* OpenLibm 0.889413 */
    -0x3.fffffffa5623708p+4588l, /* Musl 0.889413 */
    0,                         /* Apple */
    /* cos */
    -0x3.d067a048093bdf94p+9160l,/* glibc 1.50065 */
    -0x4.b0df0d1cf1e0f24p+8l,    /* icc 0.501721 */
    0,                           /* AMD */
    0,                           /* Newlib */
    0x3.e0d53885e84ce194p+4l,    /* OpenLibm 0.797817 */
    0x3.e0d53885e84ce194p+4l,    /* Musl 0.797817 */
    0,                         /* Apple */
    /* cosh */
    0x2.c5d3754804a5eb84p+12l, /* glibc 3.39084 */
    0x7.fd5d1c347e04d248p-4l,  /* icc 0.501046 */
    0,                         /* AMD */
    0,                         /* Newlib */
    0x2.c5d374f9436efd1p+12l,  /* OpenLibm 4.85758 */
    0x2.c5d37484e4c162bp+12l,  /* Musl 3.72979 */
    0,                         /* Apple */
    /* erf */
    0xd.7f1c62a10a7da4dp-4l,   /* glibc 1.15634 */
    -0x1.c5f27c68de6b5a76p-4l, /* icc 0.516833 */
    0,                         /* AMD */
    0,                         /* Newlib */
    0xd.7f1c62a10a7da4dp-4l,   /* OpenLibm 1.15634 */
    0xd.7f1c62a10a7da4dp-4l,   /* Musl 1.15634 */
    0,                         /* Apple */
    /* erfc */
    0x1.57fdf4d034d585f6p+0l,  /* glibc 4.60506 */
    0x2.f35504f2fa42e838p-4l,  /* icc 0.525990 */
    0,                         /* AMD */
    0,                         /* Newlib */
    0x1.5cc0e11b5d9aa3e4p+0l,  /* OpenLibm 5.66130 */
    0x1.5c92631d94d20a42p+0l,  /* Musl 5.03651 */
    0,                         /* Apple */
    /* exp */
    0x5.8b910ec3594e61ep-4l,   /* glibc 1.26652 */
    0x2.c590e6ab0d71c77p+12l,  /* icc 0.500644 */
    0,                         /* AMD */
    0,                         /* Newlib */
    0x8.aa15fdd49fc0238p+0l,   /* OpenLibm 1.99019 */
    -0x2.c5a107784532210cp+12l,/* Musl 1.534455698063434 */
    0,                         /* Apple */
    /* exp10 */
    0x1.228adc23322c8486p+12l, /* glibc 1.49609 */
    -0x1.2ab76ac25255a1aap+12l,/* icc 0.500307 */
    0,                         /* AMD */
    0,                         /* Newlib */
    0,                         /* OpenLibm */
    0xd.41cfea59bdcc016p+8l,   /* Musl 40.083 */
    0,                         /* Apple */
    /* exp2 */
    -0x7.3f29466b59f9019p-4l,  /* glibc 0.787028 */
    -0x3.ef2d27bf02a002bp-16l, /* icc 0.500109 */
    0,                         /* AMD */
    0,                         /* Newlib */
    -0xf.fffff464e1ae63ep-12l, /* OpenLibm 2.17787 */
    -0x7.3f29466b59f9019p-4l,  /* Musl 0.787028 */
    0,                         /* Apple */
    /* expm1 */
    0x5.8b9235e9c8acc8a8p-4l,  /* glibc 3.07429 */
    -0x1.004000b7684add34p-8l, /* icc 0.501961 */
    0,                         /* AMD */
    0,                         /* Newlib */
    0x6.61a3bd07acfccfcp-4l,   /* OpenLibm 1.93526 */
    0x2.c5c85fdf170c604cp+12l, /* Musl 9704.96 */
    0,                         /* Apple */
    /* j0 */
    -0x2.67a2a5d2e367f784p+0l, /* glibc 9.78435e+17 */
    -0x1.6a09e667f3bd238cp-32l, /* icc 0.500000000000001 */
    0,                         /* AMD */
    0,                         /* Newlib */
    0,                         /* OpenLibm */
    0,                         /* Musl */
    0,                         /* Apple */
    /* j1 */
    0x3.d4eaaeb5ede115p+0l,    /* glibc 3.37624e+18 */
    -0x1.8p-16444l,            /* icc 0.5 */
    0,                         /* AMD */
    0,                         /* Newlib */
    0,                         /* OpenLibm */
    0,                         /* Musl */
    0,                         /* Apple */
    /* lgamma */
    -0x3.ec9933945d8faa4p+0l,  /* glibc 11.9149 */
    -0x4.088ad64714fd4768p+0l, /* icc 0.547896 */
    0,                         /* AMD */
    0,                         /* Newlib */
    -0x2.74ff92c01f0d82acp+0l, /* OpenLibm 9.07174e+19 */
    -0x2.74ff92c01f0d82acp+0l, /* Musl 9.07174e+19 */
    0,                         /* Apple */
    /* log */
    0x1.20dafa7e0191b02ep+0l,  /* glibc 0.996972 */
    0x1.0ffd0d3bcf067118p+0l,  /* icc 0.500578 */
    0,                         /* AMD */
    0,                         /* Newlib */
    0x1.672126d956ceeb06p+0l,  /* OpenLibm 1.21068 */
    0x1.20dafa7e0191b02ep+0l,  /* Musl 0.996972 */
    0,                         /* Apple */
    /* log10 */
    0x1.2714ccae2ade7392p+0l,  /* glibc 1.35794 */
    0x1.01002619de32e7cp+0l,   /* icc 0.501600 */
    0,                         /* AMD */
    0,                         /* Newlib */
    0xb.ff9ef77c6e589fp-4l,    /* OpenLibm 1.19878 */
    0x1.2714ccae2ade7392p+0l,  /* Musl 1.35794 */
    0,                         /* Apple */
    /* log1p */
    -0x6.44b3c0d9d72665d8p-4l, /* glibc 2.48215 */
    -0xe.fefa23913fa3eb7p-8l,  /* icc 0.500606 */
    0,                         /* AMD */
    0,                         /* Newlib */
    -0x4.c669bd1813ec8bd8p-4l, /* OpenLibm 2.59783 */
    -0x6.44b3c0d9d72665d8p-4l, /* Musl 2.48215 */
    0,                         /* Apple */
    /* log2 */
    0x1.0596c8bb5ed5b5dep+0l,  /* glibc 0.994588 */
    0x1.01004edb8a2eb6e8p+0l,  /* icc 0.501486 */
    0,                         /* AMD */
    0,                         /* Newlib */
    0x1.6646b082fd1065cep+0l,  /* OpenLibm 1.63300 */
    0x1.0596c8bb5ed5b5dep+0l,  /* Musl 0.994588 */
    0,                         /* Apple */
    /* sin */
    -0x6.e2368c0ed74e5698p+16l,/* glibc 1.50130 */
    -0xc.141cf155623856bp+8l,  /* icc 0.501792 */
    0,                         /* AMD */
    0,                         /* Newlib */
    -0x2.a2a4aca336af4538p+8l, /* OpenLibm 0.797187 */
    -0x2.a2a4aca336af4538p+8l, /* Musl 0.797187 */
    0,                         /* Apple */
    /* sinh */
    0x2.c5d376167f4052f4p+12l, /* glibc 3.38626 */
    0x7.ad91698a9140af2p-4l,   /* icc 0.502316 */
    0,                         /* AMD */
    0,                         /* Newlib */
    -0x2.c5d375cbe7e4a81cp+12l,/* OpenLibm 4.84066 */
    0x2.c5c85fdbc1ccc354p+12l, /* Musl 9704.917289058616 */
    0,                         /* Apple */
    /* sqrt */
    0xf.fffffffffffffffp-4l,   /* glibc 0.5 */
    0xf.fffffffffffffffp-4l,   /* icc 0.5 */
    0,                         /* AMD */
    0xf.fffffffffffffffp-4l,   /* Newlib */
    0xf.fffffffffffffffp-4l,   /* OpenLibm 0.5 */
    0xf.fffffffffffffffp-4l,   /* Musl 0.5 */
    0,                         /* Apple */
    /* tan */
    0x1.974ccdb290851e7cp+8l,  /* glibc 1.74706 */
    0x3.899166e67b16dd7p+4l,   /* icc 0.503939 */
    0,                         /* AMD */
    0,                         /* Newlib */
    -0x6.fc3c72b4743ac9c8p+8l, /* OpenLibm 0.960467 */
    -0x6.fc3c72b4743ac9c8p+8l, /* Musl     0.960467 */
    0,                         /* Apple */
    /* tanh */
    0x3.b9979a543d0fbfa8p-4l,  /* glibc 3.21217 */
    0x7.fb80104d932bceep-4l,   /* icc 0.505608 */
    0,                         /* AMD */
    0,                         /* Newlib */
    -0x3.8b0575f7a027f718p-4l, /* OpenLibm 2.5499 */
    0x4.024182351388d15p-4l,   /* Musl 2.94887 */
    0,                         /* Apple */
    /* tgamma */
    -0xb.161d242d4b9282ap+0l,  /* glibc 9.54949 */
    -0x6.9c7a8fb06c63eb5p+8l,  /* icc 0.553154 */
    0,                         /* AMD */
    0,                         /* Newlib */
    -0x6.db747ae147ae148p+8l,  /* OpenLibm inf */
    -0x2.8d19fd20f3aa62cp+4l,  /* Musl 3.68935e+19 */
    0,                         /* Apple */
    /* y0 */
    0xe.4c175c6a0bf51e8p-4l,   /* glibc 1.3775e+18 */
    0x1.8a4b874528c14f1cp+2652l,/* icc 0.4999999999859818 */
    0,                         /* AMD */
    0,                         /* Newlib */
    0,                         /* OpenLibm */
    0,                         /* Musl */
    0,                         /* Apple */
    /* y1 */
    0xb.bfc89c6a1903022p+0l,   /* glibc 4.60035e+18 */
    0xd.749961e354cf884p-4284l, /* icc 0.4999999999979466 */
    0,                         /* AMD */
    0,                         /* Newlib */
    0,                         /* OpenLibm */
    0,                         /* Musl */
    0,                         /* Apple */
  };
TYPE extra[SIZE_EXTRA] = {
  0xf.fffffffffffffffp-4l
  };
#endif
#ifdef USE_FLOAT128
  TYPE worst[SIZE] = {
    /* acos */
    Q(0x9.fdbe71e81d65064f0f24b2602998p-4),    /* GNU libc  1.27700 */
    Q(0xf.f80616c2416bf63c33cd53983458p-4),    /* icc 19.1.3.304 0.501406 */
    /* acosh */
    Q(0x1.0f97586eba090200118df0902f99p+0),    /* GNU libc  3.99173 */
    Q(0x1.004ae7a1e9d7b621b12baeda616dp+0),    /* icc 19.1.3.304 0.500643 */
    /* asin */
    Q(0x7.79659a0b568bad280c8ec7eb8278p-4),   /* GNU libc  1.19480 */
    Q(0x7.ff86cc20db4e6f7fdb31fac89accp-8),   /* icc       0.501383 */
    /* asinh */
    Q(0x5.a924236647ffb723576b172b52fcp-4),    /* GNU libc 3.94073 */
    Q(0x1.0000f6bea05a0cafd1e7673e693bp-4),    /* icc 0.500541 */
    /* atan */
    Q(-0x3.6b102e0837ef34968e2c339dc6cp-4),    /* GNU libc  1.40497 */
    Q(-0x1.15eb4e54ee6ca15bfa2318ef476bp+0),   /* icc       0.500299 */
    /* atanh */
    Q(0x2.c02a24f3472c7840afbd8cfb68bap-4),    /* GNU libc  3.88862 */
    Q(-0xd.9fe29c463116c87fa567e436489p-8),    /* icc 19.1.3.304 0.500568 */
    /* cbrt */
    Q(-0xb.5096fcd4c900f68990fe999ba7fp+11884),/* GNU libc 0.735692 */
    Q(-0x2.10d29fbb2036d1d7ffbfdd99a3dcp+10912),/* icc 0.500125 */
    /* cos */
    Q(0xe.6672d458b05edf50af4fab1a42p+40),     /* GNU libc  1.51287 */
    Q(-0x6.081f6e15f81d27ac2a6323439ad8p+2232),/* icc       0.500719 */
    /* cosh */
    Q(-0x2.c5d376eefcd4bbeb000452d84662p+12),  /* GNU libc  1.91422 */
    Q(-0x2.b927c1a77ac2aced38aad663b41ap+4),   /* icc 0.500517 */
    /* erf */
    Q(0xd.f3ba21af002c4987aa75c43a0808p-4),    /* GNU libc  1.39994 */
    Q(0x5.a5182e2e3fce6963a492839ebb3cp-8),    /* icc       0.500649 */
    /* erfc */
    Q(0x1.5166e0efc44a9dfc79b8c8873a99p+0),    /* GNU libc  4.32726 */
    Q(0x6.0a5ca72c4efcc505d9b7cf9a88e8p+0),    /* icc 0.503822 */
    /* exp */
    Q(-0x2.c5b3724c8c4539ede2e1ee77419ep+12),  /* glibc 0.75001 */
    Q(-0x5.6622c128e27c6a9113094ad0fd64p-8),   /* icc 0.500444 */
    /* exp10 */
    Q(0xe.72e681b1f4f21cfe05aac3578cb8p-4),    /* GNU libc 1.99982 */
    Q(0x1.1e2a2ef09a4f66e4ec2574b62c49p+12),   /* icc 0.500475 */
    /* exp2 */
    Q(0xf.ffffa0ed8d14e72c9a27c16c32c8p-4),    /* GNU libc  1.07277 */
    Q(-0x7.cac40d04ef369e25cd005dffe05p-8),    /* icc 0.500372 */
    /* expm1 */
    Q(0x5.a1f428076faa87bb8d8482af6ad4p-4),    /* GNU libc  1.63567 */
    Q(0x8.ca3ec0644d8a8b06d14b2f7e0508p+4),    /* icc 0.500434 */
    /* j0 */
    Q(-0x8.a75ab6666f64eae68f8eb383dad8p+0),   /* GNU libc  4.10e+32 */
    Q(0x3.7c3f883498c0d5e0dab7e54a98b2p+4),    /* icc 19.1.3.304 2.89264e+28 */
    /* j1 */
    Q(0x3.d4eaaeb5ede114ff552b1726d4ep+0),     /* GNU libc  2.45301e+33 */
    Q(-0x9.4700eda0ef55f762ab92d73e27e8p+4),   /* icc 19.1.3.304 2.93445e+29 */
    /* lgamma */
    Q(-0x3.ec25f2bf2a4927e95ff5ea041c6ep+0),   /* GNU 12.9039 */
    Q(-0x3.24c1b793cb35efb8be699ad3d9bap+0),   /* icc 19.1.3.304 2.78519e+30 */
    /* log */
    Q(0xf.d01665a825be5f3c0c7d03e0c7fp-4),     /* glibc 1.04404 */
    Q(0x2.b77cdc74c184c83993bb7bca672p-4912),  /* icc 0.500099 */
    /* log10 */
    Q(0x1.6a291ea0aa11fb374f1df8b3ac6bp+0),    /* GNU libc 2.00831 */
    Q(0x1.99c491f04a5f9334ba5bb664aa62p-12364),/* icc 0.500174 */
    /* log1p */
    Q(0x6.a2681ee1c8522a86c4c7cc3cca28p-4),    /* GNU libc 3.47945 */
    Q(-0x6.2611e37be5cf438886527daadc4p-12),   /* icc 0.500299 */
    /* log2 */
    Q(0xb.54170d5cfa8fd72a47d6bda19068p-4),    /* GNU libc 3.30083 */
    Q(0xf.f63cee8e97ac6783532625273eap-4),     /* icc 0.5003182973124203 */
    /* sin */
    Q(0x5.6a5005df151cc2274e115647e9acp+64),   /* GNU libc  1.51671 */
    Q(0x4.246e3c1f108f5c75d0f326fc622p+5604),  /* icc       0.500615 */
    /* sinh */
    Q(0x6.808b50281e9909f27386f947177p-4),     /* GNU libc  2.0557 */
    Q(-0x1.6606d9a4bbc4e99f192eef24443ap+0),   /* icc 0.500733 */
    /* sqrt */
    Q(0xf.fffffffffffffffffffffffffff8p-4),    /* GNU libc  0.500 */
    Q(0xf.fffffffffffffffffffffffffff8p-4),    /* icc 19.1.3.304 0.500 */
    /* tan */
    Q(-0x3.832b771f9462df46117b6a863fa2p+8),   /* GNU libc  1.05232 */
    Q(-0x6.08aab2984de105f5078f59ed5874p+320), /* icc 19.1.3.304 0.501278 */
    /* tanh */
    Q(-0x3.c26abeca541298cca288adbd1e12p-4),   /* GNU libc 2.38002 */
    Q(-0x2.01ccac403cc83f17b47f7e4ad01ep-4),   /* icc      0.500432 */
    /* tgamma */
    Q(-0x1.7057f50b37aa8cddab347b44d58p+4),    /* GNU libc  10.4736 */
    Q(0x4.000047bf7dd56b027a0eb6672638p-15344),/* icc 19.1.3.304 8193.42 */
    /* y0 */
    Q(0x6.b99c822052e965e1754eb5ffeb08p+4),    /* GNU libc  1.68281e+33 */
    Q(0x3.9561432d16442ec543c74876d1c8p+4),    /* icc 4.7897e+27 */
    /* y1 */
    Q(0x2.3277da9bfe485c85c35e5bcc806p+0),     /* GNU libc  3.46846e+33 */
    Q(0x2.80bc307275f6a6a3feb2ab211838p+4)     /* icc 19.1.3.304 1.44943e+30 */
  };
TYPE extra[SIZE_EXTRA] = {
  0
  };
#endif
  
  int nt = omp_get_thread_num();
  TYPE * x = xpH[nt];
  double d[2*SIZE];  // more than enough
  double Dbest0;
  int i;
  /* first check the 'worst' values for the given library, with the given
     rounding mode */
  fesetround (rnd1[rnd]);
  int nval=0;
  for (i = NUMBER; i < SIZE; i+=NLIBS)
    {
      x[nval++] = worst[i];
    }
    assert(nval<2*SIZE);
    distance (x,d,nval);
    for (i=0; i<nval; ++i) {
      if (d[i] > Dbest)
        {
          Dbest = d[i];
          Xmax = x[i];
        }
    }
  Dbest0 = Dbest;
  /* then check the 'worst' values for the other libraries */
  nval=0;
  for (i = 0; i < SIZE; i++)
    {
      if ((i % NLIBS) == NUMBER)
        continue;
      x[nval++] = worst[i];
    }
   for (i = 0; i < SIZE_EXTRA; i++)
    {
      x[nval++] = extra[i];
    }
    assert(nval<2*SIZE);
    distance (x,d,nval);
    for (i=0; i<nval; ++i) {
      if (d[i] > Dbest)
        {
          Dbest = d[i];
          Xmax = x[i];
        }
    }
    printf("initial worse at %a is %a %a\n",Xmax,Dbest0,Dbest);
#endif /* WORST */

  int nthreads, n;
#pragma omp parallel
  nthreads = omp_get_num_threads ();
  if (verbose)
    printf ("Using %d threads\n", nthreads);

#pragma omp parallel for
  for (n = 0; n < nthreads; n++)
    doit (seed + n);
#ifdef WORST
  if (Dbest > Dbest0)
    printf ("NEW ");
#endif
  fesetround (FE_TONEAREST);
  printf ("%s %d %d ", NAME, mode_best, Rbest);
  print_type_hex (Xmax);
  double Dbestu = ulps (Xmax);
  printf (" [%.0f]", Dbestu);
  printf (" [");
  print_error (Dbest);
  printf ("] %.6g %.16g", Dbest, Dbest);
  printf ("\n");
  if (verbose)
    {
      mpfr_t xx, yy;
      TYPE y;
      y = wrap_foo(Xmax);
      // y = FOO2 (Xmax);
      printf ("at %a\n", Xmax);
      printf ("libm gives %a\n", y);
      mpfr_set_emax (EMAX);
      mpfr_init2 (xx, PREC);
      mpfr_init2 (yy, PREC);
      mpfr_set_type (xx, Xmax, MPFR_RNDN);
      MPFR_FOO (yy, xx, rnd);
      y = mpfr_get_type (yy, MPFR_RNDN);
      printf ("mpfr gives %a\n", y);
      mpfr_clear (xx);
      mpfr_clear (yy);
    }
  fflush (stdout);
  mpfr_free_cache ();
#ifdef STAT
  printf ("eval_heuristic=%lu eval_exhaustive=%lu\n",
          eval_heuristic, eval_exhaustive);
#endif
  return 0;
}
