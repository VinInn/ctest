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

  auto f1 = f;
  TwoFloat<float> f2(1.e-3*std::sqrt(3.f),1.e-6*std::sqrt(2.f));
  auto d1 = d;
  double d2 = double(f2.hi())+double(f2.lo());
  std::cout << std::hexfloat << f2.hi() << ',' << f2.lo() << std::endl;
  std::cout << std::hexfloat << d2 << std::endl;


  return 0;
}
