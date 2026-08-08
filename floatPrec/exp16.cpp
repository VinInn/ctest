#include "Exp16.h"
#include <iostream>



int main() {

  std::cout << std::numeric_limits<uint16_t>::max() << std::endl;
  std::cout << 5./std::numeric_limits<uint16_t>::max() << std::endl;
  std::cout << std::numeric_limits<uint16_t>::max()/5. << std::endl;

  double c = (5./std::numeric_limits<uint16_t>::max());
  std::cout << "exp " << std::exp(c) << ' ' << std::exp(ldexp(c,4)) << ' ' << std::exp(ldexp(c,8)) << ' ' << std::exp(ldexp(c,12))  << std::endl;
  Exp16  exp16(5.);


{
  Exp16::I16 u; u.i16 = 1024+19;
  std::cout << (uint16_t)(u.i4.b0) << ',' << (uint16_t)(u.i4.b1) << ',' << (uint16_t)(u.i4.b2) << ',' << (uint16_t)(u.i4.b3) << std::endl;
}

{
  int16_t eta = std::round(3.5*(std::numeric_limits<uint16_t>::max()/5.));
  std::cout << eta << ' ' << eta*(5./std::numeric_limits<uint16_t>::max()) << std::endl;
{
  Exp16::I16 u; u.i16 = std::abs(eta);
  std::cout << (uint16_t)(u.i4.b0) << ',' << (uint16_t)(u.i4.b1) << ',' << (uint16_t)(u.i4.b2) << ',' << (uint16_t)(u.i4.b3) << std::endl;
  std::cout << "exp(3.5) " << std::exp(3.5) << ' ' <<   exp16.pexp(eta) << std::endl;
  std::cout << "exp(-3.5) " << std::exp(-3.5) << ' ' <<   exp16.nexp(eta) << std::endl;
  std::cout << "sch(3.5) " << std::sinh(3.5) << ' ' << std::cosh(3.5) << std::endl;
}
}

{
  uint16_t eta = std::round(4.9*(std::numeric_limits<uint16_t>::max()/5.));
  std::cout << eta << ' ' << eta*(5./std::numeric_limits<uint16_t>::max()) << std::endl;
{
  Exp16::I16 u; u.i16 = eta;
  std::cout << (uint16_t)(u.i4.b0) << ',' << (uint16_t)(u.i4.b1) << ',' << (uint16_t)(u.i4.b2) << ',' << (uint16_t)(u.i4.b3) << std::endl;
}
}

  return 0;

}
