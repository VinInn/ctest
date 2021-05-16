#include <cstdint>
inline float roughLog(float x) {
  union IF {uint32_t i; float f;};
  IF z; z.f=x;
  uint32_t lsb = 1<21;
  z.i +=lsb;
  z.i >>= 21;
  auto f = z.i&3;
  int ex = int(z.i >>2) -127;

  const float frac[4] = {0.f,0.25f,0.5f,0.75f};
  return float(ex)+frac[f];
  

}


#include<iostream>
int main() {

  std::cout << roughLog(0.5f) << std::endl;
  std::cout << roughLog(0.25f) << std::endl;
  std::cout << roughLog(0.75f) << std::endl;


  std::cout << roughLog(4.f) << std::endl;
  std::cout << roughLog(5.f) << std::endl;
  std::cout << roughLog(6.f) << std::endl;
  std::cout << roughLog(7.f) << std::endl;
  std::cout << roughLog(8.f) << std::endl;
  std::cout << roughLog(9.f) << std::endl;

}
