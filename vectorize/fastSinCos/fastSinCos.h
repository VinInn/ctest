#include<cmath>
namespace details {
  
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

/// Single precision sincos  valid in [-pi,pi]
inline void fast_sincosf( const float xx, float & s, float &c ) {

    constexpr float PIO4F = M_PI/4.;
    constexpr float PIO2F = M_PI/2.;
    constexpr float PIF = M_PI;

    auto axx = std::abs(xx);
    auto x1 =  axx < PIO2F ?  axx : PIF - axx;
    auto x =   x1  < PIO4F ? x1 : PIO2F - x1;

    float ls,lc;
    details::fast_sincosf_m45_45(x,ls,lc);

    auto cc = x1 < PIO4F ? lc : ls;
    auto ss = x1 < PIO4F ? ls : lc;
    s = copysignf(ss,xx);

    // change sign of cos if negative
    c = axx < PIO2F ? cc : -cc;
  }


inline float fast_sinf(float x){float s,c;fast_sincosf(x,s,c);return s;}
inline float fast_cosf(float x){float s,c;fast_sincosf(x,s,c);return c;}
