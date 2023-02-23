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
  // constexpr float PIO4F = M_PI/4.;
  //auto xx = std::abs(x) < PIO4F ?  x : 0.25f*x; 
  Float ls,lc;

  sincosf0(xx,ls,lc);

  // back...
  // use poly expansion
  // http://mathworld.wolfram.com/Multiple-AngleFormulas.html
  auto ss= ls*ls;
  // s = std::abs(x) < PIO4F ? ls : 8.f*lc*ls*(0.5f-ss);
  // c = std::abs(x) < PIO4F ? lc : 1.f-8.f*ss*(0.5f+(0.5f-ss));

  // s = 8.f*lc*ls*(0.5f-ss);
  // c = 1.f-8.f*ss*(1.0f-ss);

  s = lc*ls*(4.f-8.f*ss);
  c = 1.f-ss*(8.f-8.f*ss);


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

