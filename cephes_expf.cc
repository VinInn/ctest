#include<cmath>

static float MAXNUMF = 3.4028234663852885981170418348451692544e38f;
static float MAXLOGF = 88.72283905206835f;
static float MINLOGF = -103.278929903431851103f; /* log(2^-149) */

static float MINEXP2 = 1.40129846432482e-45f; // 

static float LOGE2F = 0.693147180559945309f;
static float SQRTHF = 0.707106781186547524f;
static float PIF = 3.141592653589793238f;
static float PIO2F = 1.5707963267948966192f;
static float MACHEPF = 5.9604644775390625E-8;


static float LOG2EF = 1.44269504088896341f;

static float C1 =   0.693359375f;
static float C2 =  -2.12194440e-4f;


inline float i2f(int x) {
  union { float f; int i; } tmp;
  tmp.i=x;
 return tmp.f;
}


inline int f2i(float x) {
  union { float f; int i; } tmp;
  tmp.f=x;
 return tmp.i;
}


inline float cephes_expf(float xx) {
  float x, z;
  int n;
  
  x = xx;
  
  x =  (x > MAXLOGF) ? MAXLOGF : x;
  x =  (x < MINLOGF) ? MINLOGF : x;
  


/* Express e**x = e**g 2**n
 *   = e**g e**( n loge(2) )
 *   = e**( g + n loge(2) )
 */
  z = std::floor( LOG2EF * x + 0.5f ); /* floor() truncates toward -infinity. */
  
  x -= z * C1;
  x -= z * C2;
  n = z;
  
  z = x * x;
  /* Theoretical peak relative error in [-0.5, +0.5] is 4.2e-9. */
  z =
    ((((( 1.9875691500E-4f  * x
	  + 1.3981999507E-3f) * x
	+ 8.3334519073E-3f) * x
       + 4.1665795894E-2f) * x
      + 1.6666665459E-1f) * x
     + 5.0000001201E-1f) * z
    + x
    + 1.0;
  
  /* multiply by power of 2 */
  //  x = z * pow(2,n);
  return  z *  i2f((n+0x7f)<<23);
  // x = ldexpf( z, n );
  //  x=z;
  //return( x );
}



inline float cephes_logf( float xx ) {

  float x = xx;


  //* Test for domain
  if( x <=  MINEXP2) x= MINEXP2;


  // x = frexpf( x, &e );
  
  // exponent (-1)
  int n = f2i(x); 
  int e = n >> 23;

  // fractional part
  constexpr int inv_mant_mask =  ~0x7f800000;
  const int p05 = f2i(0.5f);
  n &= inv_mant_mask;
  n |= p05;
  x = i2f(n);
  float fe=e;
  if( x > SQRTHF ) fe+=1;
  x =   x - 1.0;
  if( x < SQRTHF )   x += x;

  float z = x * x;	 

  float y =
    (((((((( 7.0376836292E-2f * x
	     - 1.1514610310E-1f) * x
	   + 1.1676998740E-1f) * x
	  - 1.2420140846E-1f) * x
	 + 1.4249322787E-1f) * x
	- 1.6668057665E-1f) * x
       + 2.0000714765E-1f) * x
      - 2.4999993993E-1f) * x
     + 3.3333331174E-1f) * x * z;


  y += -2.12194440e-4f * fe;
  y +=  -0.5 * z;  // y - 0.5 x^2 
  
  y = x + y;   // ... + x  
  
  y += 0.693359375f * fe;
  
  
  return( y );
}



/*
void expV( float const * __restrict__ a, float * __restrict__ z) {
  for (int i=0; i!=1024; ++i)
    z[i] = cephes_expf(a[i]);

}
*/


namespace vect {
  float __attribute__ ((aligned(16))) a[1024];
  float __attribute__ ((aligned(16))) b[1024];

  void expV() {
    for (int i=0; i!=1024; ++i)
      b[i] = cephes_expf(a[i]);
    
  }

  
  void logV() {
    for (int i=0; i!=1024; ++i)
      b[i] = cephes_logf(a[i]);
    
  }
  

}
