#include <cmath>

// only for  -45deg < x < 45deg
template<typename Float>
inline void sincosf0(Float x, Float & s, Float &c ) {

  // constexpr float kk = (M_PI/8)*(M_PI/8); 
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
  auto xx = 0.25f*x;
 
  Float ls,lc;

  sincosf0(xx,ls,lc);

  // back...
  /*
  // duplicate twice
  auto ss = 2.f*ls*lc;
  auto cc = lc*lc - ls*ls;
  s = 2.f*ss*cc;
  c = cc*cc - ss*ss;
  */
  // use poly expansion
  // http://mathworld.wolfram.com/Multiple-AngleFormulas.html
  auto ss= 2.f*ls*ls;
  s = 4.f*lc*ls*(1.f-ss);
  c = 1.f-2.f*ss*(1.0f+(1.f-ss));

}


// valid only in -pi < x < pi
template<typename Float>
inline Float simpleSin(Float x) {
 Float  s, c;
 simpleSinCos(x,s,c); 
 return s;
}

// valid only in -pi < x < pi
template<typename Float>
inline Float simpleCos(Float x) {
 Float  s, c;
 simpleSinCos(x,s,c);
 return c;
}    

