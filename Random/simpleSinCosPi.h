#include <cmath>

// only for  -45deg < x < 45deg
template<typename Float>
inline 
std::tuple<float,float> sincospi0(Float x) {

  auto z = x * x;
  
  auto s =  x * (3.1415927410125732421875f + z * (-5.16771984100341796875f + z * (2.550144195556640625f + z * (-0.592480242252349853515625f)))) ;
  auto c =  1.f + z * (-4.93479156494140625f + z * (4.057690143585205078125f + z * (-1.30715453624725341796875f))) ;  
  return {s,c};
}


// valid only in -1 < x < 1
template<typename Float>
inline 
std::tuple<float,float> simpleSinCosPi( Float x) {


  // reduce to "first quadrant"
  auto xx = 0.25f*x;

  auto [ls,lc] = sincospi0(xx);

  // back...
  // use poly expansion
  // http://mathworld.wolfram.com/Multiple-AngleFormulas.html
  auto ss= ls*ls;
  // s = std::abs(x) < 0.25f ? ls : 8.f*lc*ls*(0.5f-ss);
  // c = std::abs(x) < 0.25f ? lc : 1.f-8.f*ss*(0.5f+(0.5f-ss));

  // s = 8.f*lc*ls*(0.5f-ss);
  // c = 1.f-8.f*ss*(1.0f-ss);

  auto s = lc*ls*(4.f-8.f*ss);
  auto c = 1.f-ss*(8.f-8.f*ss);
  return {s,c};

}


// valid only in -1 < x < 1
template<typename Float>
inline Float simpleSinPi(Float x) {
 auto ret = simpleSinCosPi(x); 
 return ret.first;
}

// valid only in -1 < x < 1
template<typename Float>
inline Float simpleCosPi(Float x) {
 auto ret = simpleSinCosPi(x);
 return ret.second;
}

