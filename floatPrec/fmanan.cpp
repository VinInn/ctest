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
#include <cstdlib>

int main(int argc, char** argv) {

#ifdef __FMA__
  std::cout << "hardware fma supported" << std::endl;
#endif

#ifdef FP_FAST_FMA
  std::cout << "fast fma supported" << std::endl;
#endif


  // double x = 0x1.3333333333333p+0;
  double x =  884279719003555.0; // 1.2;
  double y=x;
  if (argc>1) x=atof(argv[1]);
  if (argc>2) y=atof(argv[2]);
  auto s = foo(x,y);
  std::cout << std::hexfloat << s << std::endl;
  std::cout << std::hexfloat << foo2(x,y) << std::endl;
  std::cout << std::hexfloat << foo2(y,x) << std::endl;
  std::cout << bar(x,y) << std::endl;

}

