#include "TwoFloat.h"
#include <iostream>

int main() {

  using namespace detailsTwoFloat;

 float h = std::sqrt(2.f);
 float l = 1.e-4*std::sqrt(3.f);
 TwoFloat<float> f(h,l, fromSum);
 TwoFloat<double> d(h,l, fromSum);


  std::cout << std::hexfloat << f.hi() << ',' << f.lo() << std::endl;
  std::cout << std::hexfloat << double(f.hi())+double(f.lo())<< std::endl;
  std::cout << std::hexfloat << d.hi() << ',' << d.lo() << std::endl;

  auto f1 = f;
  TwoFloat<float> f2(-1.e-3*std::sqrt(3.f),1.e-6*std::sqrt(2.f),  fromSum);
  TwoFloat<float> f2n(1.e-3*std::sqrt(3.f),-1.e-6*std::sqrt(2.f), fromSum);
  auto d1 = double(f.hi())+double(f.lo());
  double d2 = double(f2.hi())+double(f2.lo());
  double d2n = double(f2n.hi())+double(f2n.lo());
  std::cout << std::hexfloat << f2.hi() << ',' << f2.lo() << std::endl;
  std::cout << std::hexfloat << d2 << std::endl;

  auto sf =  f1+f2;
  auto sd = d1 + d2;
  std::cout << std::hexfloat << sf.hi() << ',' << sf.lo() << std::endl;
  std::cout << std::hexfloat << double(sf.hi()) + double(sf.lo()) << std::endl;
  std::cout << std::hexfloat << sd << std::endl;
  auto sfn =  f1-f2n;
  auto sdn = d1 - d2n;
  std::cout << std::hexfloat << sfn.hi() << ',' << sfn.lo() << std::endl;
  std::cout << std::hexfloat << double(sfn.hi()) + double(sfn.lo()) << std::endl;
  std::cout << std::hexfloat << sdn << std::endl;

{
  auto mf =  f1*f2.hi();
  auto md = d1 * f2.hi();
  std::cout << std::hexfloat << f1.hi()*f2.hi() << std::endl;
  std::cout << std::hexfloat << mf.hi() << ',' << mf.lo() << std::endl;
  std::cout << std::hexfloat << md << std::endl;
}
{
  auto mf =  f1*f2;
  auto md = d1 * d2;
  std::cout << std::hexfloat << f1.hi()*f2.hi() << std::endl;
  std::cout << std::hexfloat << mf.hi() << ',' << mf.lo() << std::endl;
  std::cout << std::hexfloat << double(mf.hi()) + double(mf.lo()) << std::endl;
  std::cout << std::hexfloat << md << std::endl;
}

  return 0;
}
