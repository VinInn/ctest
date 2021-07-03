#include <cstdint>
inline float roughLog(float x) {
  union IF {uint32_t i; float f;};
  IF z; z.f=x;
  uint32_t lsb = 1<21;
  z.i +=lsb;
  z.i >>= 21;
  auto f = z.i&3;
  int ex = int(z.i >>2) -127;

  // is this faster than 0.25f*float(f)?
  const float frac[4] = {0.f,0.25f,0.5f,0.75f};
  return float(ex)+frac[f];
  
}


#include<iostream>
#include<cmath>

int main() {

  std::cout << roughLog(0.5f) << std::endl;
  std::cout << roughLog(0.25f) << std::endl;
  std::cout << roughLog(0.75f) << std::endl;


  std::cout << roughLog(1.0f) << std::endl;
  std::cout << roughLog(4.f) << std::endl;
  std::cout << roughLog(5.f) << std::endl;
  std::cout << roughLog(6.f) << std::endl;
  std::cout << roughLog(7.f) << std::endl;
  std::cout << roughLog(8.f) << std::endl;
  std::cout << roughLog(9.f) << std::endl;



  // asssume chi2 to be c1 at 1 and c2 at Xmax 
  // chi2 = c1 + (c2-c1)/ln(Xmax)*ln(x)
  // so slope = c2-c1)/ln(Xmax)
  constexpr float c1 = 0.9f;
  constexpr float c2 = 1.8;
  constexpr float Xmax = 10.f;
  constexpr float slope =  (c2-c1)/log2(Xmax);
  std::cout << " slope " << slope << std::endl;

  for (auto x=0.5f; x<12.f; x+=0.5f) 
    std::cout << x << ' ' << c1+slope*roughLog(x) << std::endl;


}
