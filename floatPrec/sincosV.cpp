#include <cmath>
#include<limits>

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

  auto xx = 0.25f*x; 
  float ls,lc;

  sincosf0(xx,ls,lc);

  /*
  // duplicate twice
  auto ss = 2.f*ls*lc;
  auto cc = lc*lc - ls*ls;
  s = 2.f*ss*cc;
  c = cc*cc - ss*ss;
  */
  // use poly expansion
  auto ss= ls*ls;
  s = 4.f*lc*ls*(1.f-2.f*ss);
  c = 1.f+8.f*ss*(ss-1.f);

}


#include<iostream>


int main() {
//   constexpr float pi4 = M_PI/4.;
//   constexpr float pi8 = M_PI/16.;

{
   float mc=0., ms=0;
   for (double xd=-M_PI; xd<(M_PI+0.01); xd+=0.01) {
     float x = xd;
     float s,c;  simpleSincos(x,s,c);
     mc = std::max(mc,std::abs(c-float(std::cos(xd))));
     ms = std::max(ms,std::abs(s-float(std::sin(xd))));
     if ( std::abs(s-std::sin(x))>1.e-5f  || std::abs(c-std::cos(x))> 1.e-5f ) std::cout << x << ' ' << x/M_PI << ' ' << s << '/' << c << ' ' << std::sin(x) << '/' << std::cos(x) << std::endl;
   }
   std::cout << "max diff w/r/t double " << mc << ' ' << ms << "   " << std::numeric_limits<float>::epsilon() << std::endl;
}

{
   float mc=0., ms=0;
   for (float xd=-M_PI; xd<(M_PI+0.01); xd+=0.01f) {
     float x = xd;
     float s,c;  simpleSincos(x,s,c);
     mc = std::max(mc,std::abs(c-float(std::cos(xd))));
     ms = std::max(ms,std::abs(s-float(std::sin(xd))));
     if ( std::abs(s-std::sin(x))>1.e-5f  || std::abs(c-std::cos(x))> 1.e-5f ) std::cout << x << ' ' << x/M_PI << ' ' << s << '/' << c << ' ' << std::sin(x) << '/' << std::cos(x) << std::endl;
   }                                                                                        
   std::cout << "max diff w/r/t float " << mc << ' ' << ms << "   " << std::numeric_limits<float>::epsilon() << std::endl;
}


{
   float mc=0., ms=0;
   for (double xd=-M_PI; xd<(M_PI+0.01); xd+=0.01) {
     float x = xd;
     float s=std::sin(x),c=std::cos(x);
     mc = std::max(mc,std::abs(c-float(std::cos(xd))));
     ms = std::max(ms,std::abs(s-float(std::sin(xd))));
     if ( std::abs(s-std::sin(x))>1.e-5f  || std::abs(c-std::cos(x))> 1.e-5f ) std::cout << x << ' ' << x/M_PI << ' ' << s << '/' << c << ' ' << std::sin(x) << '/' << std::cos(x) << std::endl;
   }
   std::cout << "max diff float double " << mc << ' ' << ms << "   " << std::numeric_limits<float>::epsilon() << std::endl;
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
