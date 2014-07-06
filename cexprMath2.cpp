#include <cmath>
#define  constexpr

namespace foo {
  static constexpr float a = std::cos(3.);
  static constexpr float c = std::exp(3.);
  static constexpr float d = std::log(3.);
  static constexpr float e = std::sinh(1.);
  static constexpr float e1 = std::asin(1.);
  static constexpr float h = std::sqrt(.1);
  static constexpr float p = std::pow(1.3,-0.75);
  static constexpr float z = std::asinh(1.2);
};



void bha(float);

void done(float x) {
  constexpr float c1= foo::c + 1 - foo::d -foo::e/2*foo::p ;
  bha(c1*x);

}
