#include<cmath>
template<typename T>
T foo(T a, T b) {
  return std::sqrt(a*a-b*b);
}

template<typename T>
T foo2(T a, T b) {
  return (a*a-b*b);
}


double bar(double a, double b) {
  return sqrt(fma(a,a,-b*b));
}

#include<iostream>

int main() {
  // double x = 0x1.3333333333333p+0;
  double x =  884279719003555.0; // 1.2;
  auto s = foo(x,x);
  std::cout << std::hexfloat << s << std::endl;
  std::cout << std::hexfloat << foo2(x,x) << std::endl;
  std::cout << bar(x,x) << std::endl;

}

