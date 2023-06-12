//  c++ -Ofast -march=skylake-avx512 sinperf.cpp -fopt-info-vec -mprefer-vector-width=512 -I../../vdt/include/
#include<cmath>
#include<cstdint>
#include<tuple>
#include <cstring>
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
inline
std::tuple<float,float> sincospi0(float x) {
  auto z = x * x;
  //  sin absolute accuracy is  27 bits
  float s =  x * (0x1.921fb4p1f + z * (-0x1.4abb7p2f + z * (0x1.464ca8p1f + z * (-0x1.2b0b88p-1f)))) ;
  // cos absolute accuracy is  24 bits
  float c =  0x1p0f + z * (-0x1.3bd39cp2f + z * (0x1.03b068p2f + z * (-0x1.4e7de8p0f))) ;
  return {s,c};
}


//  auto s =  x * (3.1415927410125732421875f + z * (-5.16771984100341796875f + z * (2.550144195556640625f + z * (-0.592480242252349853515625f)))) ;
//  auto c =  1.f + z * (-4.93479156494140625f + z * (4.057690143585205078125f + z * (-1.30715453624725341796875f))) ;


inline uint32_t f32_to_bits(float x)   { uint32_t u; memcpy(&u,&x,4); return u; }
inline float f32_from_bits(uint32_t x) { float u;    memcpy(&u,&x,4); return u; }

#include<limits>
#include<iostream>
#include<algorithm>
int main() {

  constexpr int N = 8388608; // 2^23;
  float * c = new float[N];
  float * s = new float[N];;
  float * x = new float[N];;
  x[0] = -0.25;
  double delta = 0.5/N; // 0.5f*std::numeric_limits<float>::epsilon()
  double smbits = 0;
  double cmbits = 0;
  for (int j=1; j<N; ++j) x[j] = x[j-1]+delta;
  for (int j=0; j<N; ++j) {
   auto [a,b] =  sincospi0(x[j]);
   s[j] = a; c[j]=b;
  } 

  static constexpr double pi = M_PI;
  for (int j=0; j<N; ++j) {
    smbits = std::max(smbits,std::abs(s[j]-std::sin(x[j]*pi)));
    cmbits = std::max(cmbits,std::abs(c[j]-std::cos(x[j]*pi)));
  }

  std::cout << "max bits " <<  smbits << ' ' << cmbits << ' ' << 0.5f*std::numeric_limits<float>::epsilon() <<  std::endl;
  return 0;

}
