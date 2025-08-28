#include "TwoFloat.h"
#include <iostream>

int main() {

 TwoFloat<float> f(std::sqrt(2.f),std::sqrt(0.05f));
 TwoFloat<double> d(std::sqrt(2.),std::sqrt(0.05));


  std::cout << std::hexfloat << f.hi() << ',' << f.lo() << std::endl;
  std::cout << std::hexfloat << double(f.hi())+double(f.lo())<< std::endl;
  std::cout << std::hexfloat << d.hi() << ',' << d.lo() << std::endl;

  return 0;
}
