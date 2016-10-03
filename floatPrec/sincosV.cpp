#include <cmath>

// only for  -45deg < x < 45deg
inline void sincosf0( float x, float & s, float &c ) {
    
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


// valid only in -pi < x < pi
inline void simpleSincos( float x, float & s, float &c ) {


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
 
  float ls,lc;

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


#include<iostream>


int main() {
   constexpr float pi4 = M_PI/4.;
//   constexpr float pi8 = M_PI/16.;

   for (float x=-M_PI; x<(M_PI+0.01); x+=0.01) {
     float s,c;  simpleSincos(x,s,c);
     if ( std::abs(s-std::sin(x))>1.e-5  || std::abs(c-std::cos(x))> 1.e-5 ) std::cout << x << ' ' << x/M_PI << ' ' << s << '/' << c << ' ' << std::sin(x) << '/' << std::cos(x) << std::endl;
   }


  float x[1024], s[1024];
  for (int i=0; i<1024; ++i)  x[i] = (M_PI*i)/1024;

  for (int i=0; i<1024; ++i) {
     float c;
     simpleSincos(x[i],s[i],c);
  }

  float ret=0;
  for (int i=0; i<1024; ++i) ret +=s[i];

  return ret>1;
}
