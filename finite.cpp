#include <cmath>
#include <limits>
#include <cstdlib>


#include <cstdio>
#include <iostream>

namespace vdt {
  template<typename T>
  bool isfinite(T x) { return std::abs(x) < std::numeric_limits<T>::max(); }
}


int main() {

  float none = atof("-1");
  float zero = atof("0");
  float nan = std::sqrt(none);
  float inf = none/zero;
  printf("%a %a\n",none,zero);
  printf("%a %a\n",nan,inf);
  std::cout << (std::isnan(nan) ? "NaN" : "not NaN")<< std::endl;
  std::cout << (std::isnan(inf) ? "NaN" : "not NaN")<< std::endl;
  std::cout << (std::isfinite(nan) ? "finite" : "not finite")<< std::endl;
  std::cout << (std::isfinite(inf) ? "finite" : "not finite")<< std::endl;
  std::cout << (vdt::isfinite(nan) ? "vdt-finite" : "not vdt-finite")<< std::endl;
  std::cout << (vdt::isfinite(inf) ? "vdt-finite" : "not vdt-finite")<< std::endl;

  return 0;
}
