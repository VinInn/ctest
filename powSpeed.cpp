#ifdef __MMX__
#include <mmintrin.h>
#endif

#ifdef __SSE__
#include <xmmintrin.h>
#endif

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef __SSE3__
#include <pmmintrin.h>
#endif




#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 5000


int main (int argc, char * argv[])
{
#ifdef __SSE__
  _mm_setcsr (_mm_getcsr () | 0x8040);    // on Intel, treat denormals as zero for full speed
#endif



  double r;
  double px[N], py[N];
  const double x = 0.9999999999999953371E+00;
  int i;

  r = atof (argv[1]);

  for (i = 0; i < N; i++)
    px[i] = x;

  printf ("x = %25.19e\nr = %25.19e\n", x, r );

  for (i = 0; i < N; i++)
    py[i] = pow (px[i], r);

  printf ("y = %25.19e\n", py[0]);


  return 0;
}
