#include <stdio.h>
#include <math.h>
#include <x86intrin.h>
#include <mpfr.h>

float
cr_rsqrt (float x)
{
 mpfr_t t;
 float y;
 mpfr_init2 (t, 24);
 mpfr_set_flt (t, x, MPFR_RNDN);
 mpfr_rec_sqrt (t, t, MPFR_RNDN);
 y = mpfr_get_flt (t, MPFR_RNDN);
 mpfr_clear (t);
 return y;
}

float
rsqrt (float x)
{
 float y;
 _mm_store_ss( &y, _mm_rsqrt_ss( _mm_load_ss( &x ) ) );
 return y;
}

float
ulp_error (float x)
{
 float y = rsqrt (x);
 float z = cr_rsqrt (x);
 float err = fabsf (y - z);
 int e;
 z = frexpf (z, &e);
 /* ulp(z) = 2^(e-24) */
 return ldexpf (err, 24 - e);
}

int
main()
{
 float x, err, maxerr = 0;
 x = 0.0118255867f;
 err = ulp_error (x);
 printf ("x=%la err=%f\n", x, err);
 for (x = 1.0; x < 4.0; x = nextafterf (x, x + x))
 {
   err = ulp_error (x);
   if (err > maxerr)
     printf ("x=%la err=%f\n", x, maxerr = err);
 }
}

