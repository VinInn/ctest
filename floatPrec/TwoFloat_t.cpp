#include "TwoFloat.h"
#include <iostream>

int main() {

 float h = std::sqrt(2.f);
 float l = 1.e-4*std::sqrt(3.f);
 TwoFloat<float> f(h,l);
 TwoFloat<double> d(h,l);


  std::cout << std::hexfloat << f.hi() << ',' << f.lo() << std::endl;
  std::cout << std::hexfloat << double(f.hi())+double(f.lo())<< std::endl;
  std::cout << std::hexfloat << d.hi() << ',' << d.lo() << std::endl;

  return 0;
}
