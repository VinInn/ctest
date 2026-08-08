#include <cmath>
#include <iostream>
#include <limits>
#include <cstdint>


struct Exp16 {

  struct HB {
    uint8_t b0:4;
    uint8_t b1:4;
    uint8_t b2:4;
    uint8_t b3:4;
  };
  union I16 {
    uint16_t i16;
    HB i4;
  };


  explicit Exp16(double emax) {
    double c = (emax/std::numeric_limits<uint16_t>::max());
    double ce[4] = {std::exp(c),std::exp(ldexp(c,4)),std::exp(ldexp(c,8)),std::exp(ldexp(c,12))};
    for (int i=0; i<16; ++i) {
      for (int j=0; j<4; ++j) {
        pefact[j][i]= std::pow(ce[j],i);
//      std::cout << efact[j][i] << ' ';
      }
//   std::cout << std::endl;
    }
    double nce[4] = {std::exp(-c),std::exp(ldexp(-c,4)),std::exp(ldexp(-c,8)),std::exp(ldexp(-c,12))};
    for (int i=0; i<16; ++i) {
      for (int j=0; j<4; ++j) {
        nefact[j][i]= std::pow(nce[j],i);
//      std::cout << nefact[j][i] << ' ';
      }
//    std::cout << std::endl;
    }
  }

  float pexp(uint16_t x) const {
    I16 u; u.i16 = x;
    return pefact[0][u.i4.b0]*pefact[1][u.i4.b1]*pefact[2][u.i4.b2]*pefact[3][u.i4.b3];
  }
  float nexp(uint16_t x) const {
    I16 u; u.i16 = x;
    return nefact[0][u.i4.b0]*nefact[1][u.i4.b1]*nefact[2][u.i4.b2]*nefact[3][u.i4.b3];
  }


float pefact[4][16];
float nefact[4][16];

};





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
