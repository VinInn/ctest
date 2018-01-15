#include<cmath>
namespace details {
  const float DP1F = 0.78515625;
  const float DP2F = 2.4187564849853515625e-4;
  const float DP3F = 3.77489497744594108e-8;
  
  const float T24M1 = 16777215.;
  const float ONEOPIO4F = 4./M_PI;

//------------------------------------------------------------------------------
/// Reduce to 0 to 45
inline float reduce2quadrant(float x, int & quad) {
    /* make argument positive */
    x = fabs(x);

    quad = int (ONEOPIO4F * x); /* integer part of x/PIO4 */

    quad = (quad+1) & (~1);
    const float y = float(quad);
    // quad &=4;
    // Extended precision modular arithmetic
    return ((x - y * DP1F) - y * DP2F) - y * DP3F;
  }
  
  
//------------------------------------------------------------------------------



/// Sincos only for -45deg < x < 45deg
inline void fast_sincosf_m45_45( const float x, float & s, float &c ) {

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

//------------------------------------------------------------------------------

} // end details namespace

/// Single precision sincos
inline void fast_sincosf( const float xx, float & s, float &c ) {
	

    int j;
    const float x = details::reduce2quadrant(xx,j);
    int signS = (j&4); 

    j-=2;

    const int signC = (j&4);
    const int poly = j&2;

    float ls,lc;
    details::fast_sincosf_m45_45(x,ls,lc);

    //swap
    if( poly==0 ) {
      const float tmp = lc;
      lc=ls; ls=tmp;
    }

    if(signC == 0) lc = -lc;
    if(signS != 0) ls = -ls;
    if (xx<0)  ls = -ls;
    c=lc;
    s=ls;
  }


inline float fast_sinf(float x){float s,c;fast_sincosf(x,s,c);return s;}
inline float fast_cosf(float x){float s,c;fast_sincosf(x,s,c);return c;}
