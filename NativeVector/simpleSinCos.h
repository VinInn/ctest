#include <cmath>

// only for  -45deg < x < 45deg
template<typename Float>
inline void sincosf0(Float x, Float & s, Float &c ) {
    
    auto z = x * x;
    
    s = (((-1.9515295891E-4f * z
	   + 8.3321608736E-3f) * z
	  - 1.6666654611E-1f) * z * x)
      + x;
    
    
    c = ((  2.443315711809948E-005f * z
	    - 1.388731625493765E-003f) * z
	 + 4.166664568298827E-002f) * z * z
      - 0.5f * z + 1.0f;
    
}


// valid only in -pi < x < pi
template<typename Float>
inline void simpleSinCos( Float x, Float & s, Float &c ) {


   // reduce to "first quadrant"

   /*  auto vectorize only in gcc 7
   constexpr float pi4 = M_PI/4.;
   constexpr float pi2 = M_PI/2.;
   auto g1 = x > pi4;
   auto xx = x;
   xx = g1 ? xx-pi2 : xx;
   auto g2 = xx > pi4;
   xx = g2 ? xx-pi2 : xx;
   */

   constexpr float pi  = M_PI;
   constexpr float pi4 = M_PI/4.;
   constexpr float pi34 = 3.*M_PI/4.;
   constexpr float pi2 = M_PI/2.;
   auto g0 = x > 0;
   auto ax = g0 ? x : -x;
   auto g1 = ax > pi4;
   auto xx = g1 ? ax-pi2 : ax;
   auto g2 = ax > pi34;
   xx = g2 ? ax-pi : xx;
 
  Float ls,lc;

  sincosf0(xx,ls,lc);

  auto sw =  g1 & (!g2);
  auto ss = sw ? lc : ls;
  auto cc = sw ? -ls : lc;
  cc = g2 ? -cc : cc;
  ss = g2 ? -ss : ss;

  // cc = g0 ? cc : -cc;
  ss = g0 ? ss : -ss;
 
  c = cc; s = ss;
}


