#include <cmath>

// only for  -45deg < x < 45deg
template<typename Float>
inline void sincospi0(Float x, Float & s, Float &c ) {

  auto z = x * x;
  
  s =  x * (3.1415927410125732421875f + z * (-5.16771984100341796875f + z * (2.550144195556640625f + z * (-0.592480242252349853515625f)))) ;
  c =  1.f + z * (-4.93479156494140625f + z * (4.057690143585205078125f + z * (-1.30715453624725341796875f))) ;  

}


// valid only in -1 < x < 1
template<typename Float>
inline void simpleSinCosPi( Float x, Float & s, Float &c ) {


  // reduce to "first quadrant"
  auto xx = 0.25f*x;
  Float ls,lc;

  sincospi0(xx,ls,lc);

  // back...
  // use poly expansion
  // http://mathworld.wolfram.com/Multiple-AngleFormulas.html
  auto ss= ls*ls;
  // s = std::abs(x) < 0.25f ? ls : 8.f*lc*ls*(0.5f-ss);
  // c = std::abs(x) < 0.25f ? lc : 1.f-8.f*ss*(0.5f+(0.5f-ss));

  // s = 8.f*lc*ls*(0.5f-ss);
  // c = 1.f-8.f*ss*(1.0f-ss);

  s = lc*ls*(4.f-8.f*ss);
  c = 1.f-ss*(8.f-8.f*ss);


}


// valid only in -1 < x < 1
template<typename Float>
inline Float simpleSinPi(Float x) {
 Float  s, c;
 simpleSinCos(x,s,c); 
 return s;
}

// valid only in -1 < x < 1
template<typename Float>
inline Float simpleCospi(Float x) {
 Float  s, c;
 simpleSinCosPi(x,s,c);
 return c;
}    

