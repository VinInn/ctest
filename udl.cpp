#include<iostream>
#include<sstream>


constexpr long double operator"" _degrees(long double d) { return d * 0.0175; }

// double operator"" _degrees(const char * d) { return atof(d) * 0.0175; }


int main() {
  double pi = 180_degrees;
  std::cout << pi << std::endl;
  return 0;
}
