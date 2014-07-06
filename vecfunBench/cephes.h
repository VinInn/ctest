#ifndef cephes_H
#define cephes_H
#include<cmath>
#include<limits>

// CEPHES_LIMITED_RANGE limits the range in input
// CEPHES_NORMALIZE  check output


namespace cephes {
  namespace cephes_details {
    static const float DP1 = 0.78515625;
    static const float DP2 = 2.4187564849853515625e-4;
    static const float DP3 = 3.77489497744594108e-8;
    static const float lossth = 8192.;
    static const float T24M1 = 16777215.;
    static const float FOPI = 1.27323954473516;
    static const float PIO4F = 0.7853981633974483096;
    static const float PIF = 3.141592653589793238;
    static const float PIO2F = 1.5707963267948966192;


    static const float MAXNUMF = 3.4028234663852885981170418348451692544e38f;
    static const float MAXLOGF = 88.72283905206835f;
    static const float MINLOGF = -88.f;
  
    //  static const float MINLOGF = -103.278929903431851103f; /* log(2^-149) */
    
    static const float MINEXP2 = 1.40129846432482e-45f; // use num lim

    static const float LOGE2F = 0.693147180559945309f;
    static const float SQRTHF = 0.707106781186547524f;
    static const float MACHEPF = 5.9604644775390625E-8;

    static const float LOG2EF = 1.44269504088896341f;
 
    static const float C1 =   0.693359375f;
    static const float C2 =  -2.12194440e-4f;

    static const double MAXLOGd =  7.08396418532264106224E2;     /* log 2**1022 */
    static const double MINLOGd = -7.08396418532264106224E2;     /* log 2**-1022 */
    static const double LOG2Ed  =  1.4426950408889634073599;     /* 1/log(2) */
    static const double C1d = 6.93145751953125E-1;
    static const double C2d = 1.42860682030941723212E-6;
    static const double SQRTH =0.70710678118654752440;
    


  }
  
  
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
  

  /* reduce to  0 < x < 45deg, return also octant 0=< oct =< 7)
   */
  inline float reduce2octant(float xx, int & oct) {
    using namespace cephes_details;
    /* make argument positive */
    float x = fabs(xx);
    
#ifdef CEPHES_LIMITED_RANGE
    x =  (x > T24M1) ?   T24M1 : x;
#endif   

    oct = FOPI * x; /* integer part of x/PIO4 */
    
    float y = ((oct+1)&(~1));
    oct &=7;
    if  (xx<0) oct = 7 - oct;
    
    // Extended precision modular arithmetic
    // return x - y * PIO4F;
    x = ((x - y * DP1) - y * DP2) - y * DP3;
    return (xx>0) ? x : -x;
  }
  
  /* reduce to  -45deg < x < 45deg, return also quadrant 0=< quad =< 3)
   */
  inline float reduce2quadrant(float x, int & quad) {
    using namespace cephes_details;
    /* make argument positive */
    x = fabs(x);
    
#ifdef CEPHES_LIMITED_RANGE
    x =  (x > T24M1) ?   T24M1 : x;
#endif
    
    quad = FOPI * x; /* integer part of x/PIO4 */
    
    quad = (quad+1) & (~1);
    float y = quad;
    // quad &=4;
    // Extended precision modular arithmetic
    return ((x - y * DP1) - y * DP2) - y * DP3;
  }
  
  
  // only for  -45deg < x < 45deg
  inline void sincosf0( float x, float & s, float &c ) {
    using namespace cephes_details;
    
    float z = x * x;
    
    s = (((-1.9515295891E-4f * z
	   + 8.3321608736E-3f) * z
	  - 1.6666654611E-1f) * z * x)
      + x;
    
    
    c = ((  2.443315711809948E-005f * z
	    - 1.388731625493765E-003f) * z
	 + 4.166664568298827E-002f) * z * z
      - 0.5f * z + 1.0f;
    
  }
  
  inline void sincosf( float xx, float & s, float &c ) {
    float ls,lc;
    int j=0;
    float x = reduce2quadrant(xx,j);
    int signS = (j&4); 
    
    j-=2;
    
    int signC = (j&4);
    
    int poly = j&2;
    
    sincosf0(x,ls,lc);
    
    //swap
    if( poly==0 ) {
      float tmp = lc;
      lc=ls; ls=tmp;
    }
    
    if(signC == 0) lc = -lc;
    if(signS != 0) ls = -ls;
    if (xx<0)  ls = -ls;
    c=lc;
    s=ls;
  }


  inline float atanf( float xx ) {
    using namespace cephes_details;
    
    float x, x0, y=0.0f, z;
    
    x = x0 = fabs(xx);
    
    
    /* range reduction */
    
    if( x0 > 0.4142135623730950f ) // * tan pi/8 
      {
	x = (x0-1.0f)/(x0+1.0f);
      }
    
    if( x0 > 2.414213562373095f )  // tan 3pi/8
      {
	x = -( 1.0f/x0 );
      }

   
     if( x0 > 0.4142135623730950f ) // * tan pi/8
      {
	y = PIO4F;
      }
      if( x0 > 2.414213562373095f )  //* tan 3pi/8
      {
	y = PIO2F;
      }    
   

    z = x * x;
    y +=
      ((( 8.05374449538e-2f * z
	  - 1.38776856032E-1f) * z
	+ 1.99777106478E-1f) * z
       - 3.33329491539E-1f) * z * x
      + x;
    
    return xx < 0 ? -y : y;
  }


  inline float atan2f( float y, float x ) {
    using namespace cephes_details;
    // move in first octant
    float xx = fabs(x);
    float yy = fabs(y);
    float tmp =0.0f;
    if (yy>xx) {
      tmp = yy;
      yy=xx; xx=tmp;
    }
    float t=yy/xx;
    float z=t;
    if( t > 0.4142135623730950f ) // * tan pi/8 
      {
	z = (t-1.0f)/(t+1.0f);
      }

    //printf("%e %e %e %e\n",yy,xx,t,z);
    float z2 = z * z;
    float ret =
      ((( 8.05374449538e-2f * z2
	  - 1.38776856032E-1f) * z2
	+ 1.99777106478E-1f) * z2
       - 3.33329491539E-1f) * z2 * z
      + z;

    // move back in place
    if (y==0) ret=0.0f;
    if( t > 0.4142135623730950f ) ret += PIO4F;
    if (tmp!=0) ret = PIO2F - ret;
    if (x<0) ret = PIF - ret;
    if (y<0) ret = -ret;
    
    return ret;

  }


  inline float expf(float xx) {
    using namespace cephes_details;
    float x, z;
    int n;
    
    x = xx;
    
#ifdef CEPHES_LIMITED_RANGE
    x =  (x > MAXLOGF) ? MAXLOGF : x;
    x =  (x < MINLOGF) ? MINLOGF : x;
#endif   
    
    
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
      + 1.0f;
  
    /* multiply by power of 2 */
    z *=  i2f((n+0x7f)<<23);

    // #ifdef CEPHES_NORMALIZE
    if (xx > MAXLOGF) z=std::numeric_limits<float>::infinity();
    if (xx < MINLOGF) z=0.f;
    //#endif

    return z;
    
  }
  
  
  
  inline float logf( float xx ) {
    using namespace cephes_details;

    float x = xx;
    
    
    //* Test for domain
#ifdef CEPHES_LIMITED_RANGE
    if( x <=  MINEXP2) x= MINEXP2;
#endif    
    
    // x = frexpf( x, &e );
    
    // exponent (-1)
    int n = f2i(x); 
    int e = (n >> 23)-127;
    
    // fractional part
    constexpr int inv_mant_mask =  ~0x7f800000;
    // const int p05 = f2i(0.5f);
    constexpr int p05 = 0x3f000000;
    n &= inv_mant_mask;
    n |= p05;
    x = i2f(n);
    float fe=e;
    if( x > SQRTHF ) fe+=1.f;
    if( x < SQRTHF )   x += x;
     x =   x - 1.0f;
   
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
    y +=  -0.5f * z;  // y - 0.5 x^2 
    
    y = x + y;   // ... + x  
    
    y += 0.693359375f * fe;
    
    
#ifdef CEPHES_NORMALIZE
    if (x <  MINEXP2) y=-std::numeric_limits<float>::infinity();
#endif

    return y;
  }
  

  inline float sinhf( float x) {
    float z = std::abs(x);
    z = expf(z);
    z = 0.5f*z - (0.5f/z);
    return ( x > 0 ) ? z : -z;
  }
  
  // valid for x<1.
  inline float sinhf0( float x) {
    float z = x * x;
    z =
      (( 2.03721912945E-4f * z
	 + 8.33028376239E-3f) * z
       + 1.66667160211E-1f) * z * x
      + x;
    return z;
  }
  
  inline float asinhf( float xx ) {
    float x = std::abs(xx);
    float z = x * x;
    z = std::sqrt( z + 1.0f );
    z = logf( x + z );
    return ( xx > 0 ) ? z : -z;
  }
  
  // valid for x<0.5
  inline float asinhf0( float xx ) {
    float x = std::abs(xx);
    float z = x * x;
    z =
      ((( 2.0122003309E-2f * z
	  - 4.2699340972E-2f) * z
	+ 7.4847586088E-2f) * z
       - 1.6666288134E-1f) * z * x
      + x;
    return ( xx > 0 ) ? z : -z;
  }
  


  inline double ll2d(unsigned long long x) {
    union { double f; unsigned long long i; } tmp;
    tmp.i=x;
    return tmp.f;
  }
  
  
  inline unsigned long long d2ll(double x) {
    union { double f; unsigned long long i; } tmp;
    tmp.f=x;
    return tmp.i;
  }
  
  inline double exp(double x) {
    using namespace cephes_details;

    /* n = round(x / log 2) */
    // a = LOG2E * x + 0.5;
    // n = (int)a;
    // n -= (a< 0);
    
    /* x -= n * log2 */
    // px = (double)n;
    
    double px = std::floor( LOG2Ed * x + 0.5 );
    
    /* x -= n * log2 */
    int n = px;
    
    x -= px * C1d;
    x -= px * C2d;
    double xx = x * x;
    
    /* px = x * P(x**2). */
    px = 1.26177193074810590878E-4;
    px *= xx;
    px += 3.02994407707441961300E-2;
    px *= xx;
    px += 9.99999999999999999910E-1;
    px *= x;
    
    /* Evaluate Q(x**2). */
    double qx = 3.00198505138664455042E-6;
    qx *= xx;
    qx += 2.52448340349684104192E-3;
    qx *= xx;
    qx += 2.27265548208155028766E-1;
    qx *= xx;
    qx += 2.00000000000000000009E0;
    
    /* e**x = 1 + 2x P(x**2)/( Q(x**2) - P(x**2) ) */
    x = px / (qx - px);
    x = 1.0 + 2.0 * x;
    
    /* Build 2^n in double. */
    n += 1023;
    // u.s[3] = (unsigned short)((n<< 4) & 0x7FF0);
    // u.i[1] =  n <<20;
    
    return x * ll2d( (unsigned long long)(n) <<52);
    
  }

  inline
  double log(double x){  
    using namespace cephes_details;
    
    double input_x=x;
    
    double y, z;
    
    double px,qx;
    
    /* separate mantissa from exponent */
    /* Note, we shoudl use frexp is used so that denormal numbers
     * will be handled properly.
     */
    
    unsigned long long n = d2ll(x);
    
    unsigned long long le = ((n >> 52) & 0x7ffL);
    int e = le;
    double fe =(e-1023);
    n &=0xfffffffffffffLL;
    constexpr unsigned long long p05 = (1022LL<<52); // d2ll(0.5);
    // const unsigned long long p05 = d2ll(0.5);  // FIXME
    n |= p05;
    x = ll2d(n);
    if( x > SQRTH ) fe+=1.;
    if( x < SQRTH )   x += x;
    x =   x - 1.0;
    
    
    /* logarithm using log(x) = z + z**3 P(z)/Q(z),
     * where z = 2(x-1)/x+1)
     */
    // not worth
    
    /* logarithm using log(1+x) = x - .5x**2 + x**3 P(x)/Q(x) */
    
    
    
    /* rational form */
    
    z = x*x;
    px =  1.01875663804580931796E-4;
    px *= x;    
    px += 4.97494994976747001425E-1;
    px *= x;    
    px += 4.70579119878881725854E0;
    px *= x; 
    px += 1.44989225341610930846E1;
    px *= x; 
    px += 1.79368678507819816313E1;
    px *= x;
    px += 7.70838733755885391666E0;
    //
    //for the final formula
    px *= x; 
    px *= z;
    
    
    qx = x;
    qx += 1.12873587189167450590E1;
    qx *=x;
    qx += 4.52279145837532221105E1;
    qx *=x;    
    qx += 8.29875266912776603211E1;
    qx *=x;    
    qx += 7.11544750618563894466E1;
    qx *=x;    
    qx += 2.31251620126765340583E1;
    
    
    y = px / qx ;
    
    y -= fe * 2.121944400546905827679e-4; 
    y -= 0.5 * z  ;
    
    z = x + y;
    z += fe * 0.693359375;
    
#ifdef CEPHES_NORMALIZE
    if (input_x > 5e307)
      z = std::numeric_limits<double>::infinity();
    if (input_x < 5e-307)
      z =  - std::numeric_limits<double>::infinity();       
#endif    
    
    return z;  
    
  }
  
  
}


#endif
  
