#include<cmath>
static float DP1 = 0.78515625;
static float DP2 = 2.4187564849853515625e-4;
static float DP3 = 3.77489497744594108e-8;
static float lossth = 8192.;
static float T24M1 = 16777215.;
static float FOPI = 1.27323954473516;
static float PIO4F = 0.7853981633974483096;
static float PIF = 3.141592653589793238;
static float PIO2F = 1.5707963267948966192;

#include<cstdio>

inline unsigned long long int rdtsc() {
  unsigned long long int x;
  __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
  return x;
}


static float sincof[] = {
  -1.9515295891E-4,
  8.3321608736E-3,
  -1.6666654611E-1
};
static float coscof[] = {
  2.443315711809948E-005,
  -1.388731625493765E-003,
  4.166664568298827E-002
};

float cephes_atanf( float xx )
{
float x, y, z;
int sign;

x = xx;

/* make argument positive and save the sign */
if( xx < 0.0f )
	{
	sign = -1;
	x = -xx;
	}
else
	{
	sign = 1;
	x = xx;
	}
/* range reduction */
if( x > 2.414213562373095f )  /* tan 3pi/8 */
	{
	y = PIO2F;
	x = -( 1.0f/x );
	}

else if( x > 0.4142135623730950f ) /* tan pi/8 */
	{
	y = PIO4F;
	x = (x-1.0f)/(x+1.0f);
	}
else
	y = 0.0;

z = x * x;
y +=
((( 8.05374449538e-2f * z
  - 1.38776856032E-1f) * z
  + 1.99777106478E-1f) * z
  - 3.33329491539E-1f) * z * x
  + x;

if( sign < 0 )
	y = -y;

return( y );
}


float cephes_atan2f( float y, float x )
{
float z, w;
int code;


code = 0;

if( x < 0.0 )
	code = 2;
if( y < 0.0 )
	code |= 1;

if( x == 0.0 )
	{
	if( code & 1 )
		{
		return( -PIO2F );
		}
	if( y == 0.0 )
		return( 0.0 );
	return( PIO2F );
	}

if( y == 0.0 )
	{
	if( code & 2 )
		return( PIF );
	return( 0.0 );
	}


switch( code )
	{
	default:
	case 0:
	case 1: w = 0.0; break;
	case 2: w = PIF; break;
	case 3: w = -PIF; break;
	}

z = atanf( y/x );

return( w + z );
}


float cephes_sinf( float xx )
{
  float *p;
  float x, y, z;
  register unsigned long j;
  register int sign;

  sign = 1;
  x = xx;
  if( xx < 0 )
    {
      sign = -1;
      x = -xx;
    }
  if( x > T24M1 )
    {
      //mtherr( "sinf", TLOSS );
      return(0.0);
    }
  j = FOPI * x; /* integer part of x/(PI/4) */
  y = j;
  /* map zeros to origin */
  if( j & 1 )
    {
      j += 1;
      y += 1.0;
    }
  j &= 7; /* octant modulo 360 degrees */
  /* reflect in x axis */
  if( j > 3)
    {
      sign = -sign;
      j -= 4;
    }
  if( x > lossth )
    {
      //mtherr( "sinf", PLOSS );
      x = x - y * PIO4F;
    }
  else
    {
      /* Extended precision modular arithmetic */
      x = ((x - y * DP1) - y * DP2) - y * DP3;
    }
  /*einits();*/
  z = x * x;
  //printf("my_sinf: corrected oldx, x, y = %14.10g, %14.10g, %14.10g\n", oldx, x, y);
  if( (j==1) || (j==2) )
    {
      /* measured relative error in +/- pi/4 is 7.8e-8 */
      /*
        y = ((  2.443315711809948E-005 * z
        - 1.388731625493765E-003) * z
        + 4.166664568298827E-002) * z * z;
      */
      p = coscof;
      y = *p++;
      y = y * z + *p++;
      y = y * z + *p++;
      y *= z; y *= z;
      y -= 0.5 * z;
      y += 1.0;
    }
  else
    {
      /* Theoretical relative error = 3.8e-9 in [-pi/4, +pi/4] */
      /*
        y = ((-1.9515295891E-4 * z
        + 8.3321608736E-3) * z
        - 1.6666654611E-1) * z * x;
        y += x;
      */
      p = sincof;
      y = *p++;
      y = y * z + *p++;
      y = y * z + *p++;
      y *= z; y *= x;
      y += x;
    }
  /*einitd();*/
  //printf("my_sinf: j=%d result = %14.10g * %d\n", j, y, sign);
  if(sign < 0)
    y = -y;
  return( y);
}


float cephes_cosf( float xx )
{
  float x, y, z;
  int j, sign;

  /* make argument positive */
  sign = 1;
  x = xx;
  if( x < 0 )
    x = -x;

  if( x > T24M1 )
    {
      //mtherr( "cosf", TLOSS );
      return(0.0);
    }

  j = FOPI * x; /* integer part of x/PIO4 */
  y = j;
  /* integer and fractional part modulo one octant */
  if( j & 1 )	/* map zeros to origin */
    {
      j += 1;
      y += 1.0;
    }
  j &= 7;
  if( j > 3)
    {
      j -=4;
      sign = -sign;
    }

  if( j > 1 )
    sign = -sign;

  if( x > lossth )
    {
      //mtherr( "cosf", PLOSS );
      x = x - y * PIO4F;
    }
  else
    /* Extended precision modular arithmetic */
    x = ((x - y * DP1) - y * DP2) - y * DP3;

  //printf("xx = %g -> x corrected = %g sign=%d j=%d y=%g\n", xx, x, sign, j, y);

  z = x * x;

  if( (j==1) || (j==2) )
    {
      y = (((-1.9515295891E-4f * z
             + 8.3321608736E-3f) * z
            - 1.6666654611E-1f) * z * x)
        + x;
    }
  else
    {
      y = ((  2.443315711809948E-005f * z
              - 1.388731625493765E-003f) * z
           + 4.166664568298827E-002f) * z * z;
      y -= 0.5 * z;
      y += 1.0;
    }
  if(sign < 0)
    y = -y;
  return( y );
}


namespace vect {


  inline float cephes_sincosf( float xx, float & s, float &c ) {
    float x, y, y0, y1, z;
    int j, j0, j1;
    float poly, sign0, sign1;
    
    /* make argument positive */
    
    x = fabs(xx);
    
    x =  (x > T24M1) ?   T24M1 : x;
    
    j = FOPI * x; /* integer part of x/PIO4 */
    
    j = (j+1) & (~1);
    y = j;
    
    j1 = (j&4);
    sign1 = j1;
    
    
    j-=2;

    j0 = (j&4);
    sign0 = j0;
   
    
    poly = (j&2);
    
  
    // Extended precision modular arithmetic
    x = ((x - y * DP1) - y * DP2) - y * DP3;
    
    //printf("xx = %g -> x corrected = %g sign=%d j=%d y=%g\n", xx, x, sign, j, y);
    
    z = x * x;
    
    y0 = (((-1.9515295891E-4f * z
	    + 8.3321608736E-3f) * z
	   - 1.6666654611E-1f) * z * x)
      + x;
    
    
    y1 = ((  2.443315711809948E-005f * z
	     - 1.388731625493765E-003f) * z
	  + 4.166664568298827E-002f) * z * z
      - 0.5f * z + 1.0f;
    
    //swap
    if( poly!=0 ) {
      float tmp = y0;
      y0=y1; y1=tmp;
    }

      

    if(sign0 == 0) y0 = -y0;
    c = y0;
    if(sign1 == 0) y1 = -y1;
    if (xx<0)  y1 = -y1;
    s = -y1;
  }
  


  inline float cephes_atanf( float xx ) {
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


  inline float cephes_atan2f( float y, float x ) {
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
    // if (y==0) ret=0.0f;
    if( t > 0.4142135623730950f ) ret += PIO4F;
    if (tmp!=0) ret = PIO2F - ret;
    if (x<0) ret = PIF - ret;
    if (y<0) ret = -ret;
    
    return ret;

  }

    /*
  inline float cephes_atan2f( float y, float x ) {
    float z=0.0f, w=0.0f;

    if ( x< 0.0f )  w = PIF; 
     
    if (y< 0.0 ) w= -PIF; 
    if ( x> 0.0f )  w =0;
   
    z = cephes_atanf( y/x );
    
    if( x == 0.0f ) z = PIO2F;


    return w+z;
  }
    */

}


float __attribute__ ((aligned(16))) a0[1024];
float __attribute__ ((aligned(16))) s0[1024];
float __attribute__ ((aligned(16))) c0[1024];

float __attribute__ ((aligned(16))) a[1024];
float __attribute__ ((aligned(16))) s[1024];
float __attribute__ ((aligned(16))) c[1024];

void scS() {
  for (int i=0; i!=1024; ++i) {
    c0[i] = cephes_cosf(a0[i]);  
    s0[i] = cephes_sinf(a0[i]);  
  }
}


void aS() {
  for (int i=0; i!=1024; ++i) {
    float z = i;
    a0[i] = cephes_atan2f(z*s0[i],z*c0[i]);  
  }
}

/*
void cV() {
  for (int i=0; i!=1024; ++i)
    c[i] = vect::cephes_cosf(a0[i]);  
}
void sV() {
  for (int i=0; i!=1024; ++i)
    s[i] = vect::cephes_sinf(a0[i]);  
}
*/

void scV() {
  for (int i=0; i!=1024; ++i)
    vect::cephes_sincosf(a0[i], s[i], c[i]);
}

void aV() {
  for (int i=0; i!=1024; ++i) {
    float z = i;
    a[i] = vect::cephes_atan2f(s0[i],c0[i]);  
  }
}



#include<cstdio>
int main() {
  
  a0[0] = -10.f*PIF;
  for (int i=1; i!=1024; ++i)
    a0[i] = a0[i-1] + 0.02*PIF;

  auto t1 =  rdtsc();
  scS();
  auto tscS =  rdtsc() -t1;
  

  
  t1 =  rdtsc();
  scV();
  auto tscV =  rdtsc() -t1;
  
  for (int i=0; i!=1024; ++i)
    if (fabs(s0[i]-s[i])>2e-07 || fabs(c0[i]-c[i])>2e-07)
      // if (fabs(b[i])!=fabs(c[i]))
      printf("%d %e %e %e %e %e\n", i,a0[i],s0[i],s[i], c0[i],c[i]);
  
  
  
  

  t1 =  rdtsc();
  aS();
  auto taS =  rdtsc() -t1;
  
  t1 =  rdtsc();
  aV();
  auto taV =  rdtsc() -t1;


  for (int i=0; i!=1024; ++i)
       if (fabs(a[i]-a0[i])>2.e-07)
      printf("%d %e %e %e\n", i, a0[i],a[i],a0[i]-a[i]);


  printf("%d %d   %d %d\n",tscS,tscV,taS,taV);

  return 0;
}
