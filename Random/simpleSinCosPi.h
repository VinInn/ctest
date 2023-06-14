#include <cmath>
#include<tuple>

// valid only for  -0.25 < x < 0.25 
/* polymomial produced using the following Sollya script
s=fpminimax(sin(pi*x), [|1,3,5,7|],[|23...|], [0;1/4], floating, relative);
c=fpminimax(cos(pi*x), [|0,2,4,6|], [|1,23...|], [0;1/4], floating, relative);
display=decimal;
acc=floor(-log2(sup(supnorm(s, sin(pi*x), [0;1/4], absolute, 2^(-40)))));
print( "// sin absolute accuracy is ",  acc, "bits" );
acc=floor(-log2(sup(supnorm(c, cos(pi*x), [0;1/4], absolute, 2^(-40)))));
print( "// cos absolute accuracy is ",  acc, "bits" );
display=hexadecimal;
print("   float s = ", horner(s) , ";");
print("   float c = ", horner(c) , ";");
*/
template<typename Float>
inline 
std::tuple<Float,Float> sincospi0(Float x) {
  auto z = x * x;
  //  sin absolute accuracy is  27 bits
  Float s =  x * (0x1.921fb4p1f + z * (-0x1.4abb7p2f + z * (0x1.464ca8p1f + z * (-0x1.2b0b88p-1f)))) ;
  // cos absolute accuracy is  24 bits
  Float c =  0x1p0f + z * (-0x1.3bd39cp2f + z * (0x1.03b068p2f + z * (-0x1.4e7de8p0f))) ;
  return {s,c};
}


//  auto s =  x * (3.1415927410125732421875f + z * (-5.16771984100341796875f + z * (2.550144195556640625f + z * (-0.592480242252349853515625f)))) ;
//  auto c =  1.f + z * (-4.93479156494140625f + z * (4.057690143585205078125f + z * (-1.30715453624725341796875f))) ;


//return sin, cos for a 4x angle
template<typename Float>
inline void sinCosX4(Float & s, Float & c) {
  auto ls = s; auto lc = c;

  // use poly expansion
  // http://mathworld.wolfram.com/Multiple-AngleFormulas.html
  auto ss= ls*ls;

  // s = 8.f*lc*ls*(0.5f-ss);
  // c = 1.f-8.f*ss*(1.0f-ss);

  s = lc*ls*(4.f-8.f*ss);
  c = 1.f-ss*(8.f-8.f*ss);

}


// valid only in -1 < x < 1
template<typename Float>
inline 
std::tuple<float,float> simpleSinCosPi( Float x) {
  // reduce to "first quadrant"
  auto xx = 0.25f*x;

  auto ret = sincospi0(xx);
  sinCosX4(ret.first,ret.second); 
  // back...
  return ret;
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

