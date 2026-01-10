#include <cmath>
#include <cstdio>
#include <iostream>
#include <cstdint>


// better b be positive and <1
inline double fastPow(double a, double b) {
  union {
    double d;
    int x[2];
  } u = { a };
  u.x[1] = (int)(b * (u.x[1] - 1072632447) + 1072632447);
  u.x[0] = 0;
  return u.d;
}

inline float fastPow(float a, float b) {
  union {
    float d;
    uint32_t x;
  } u = { a };
  u.x = (uint32_t)(b * (u.x - 1064866808) + 1064866808);  // 0.971007823944
  return u.d;
}

inline float fastPow1(float a, float b) {
  auto x0 = fastPow(a,b);
  return x0*(1.f-b*(1.f-a/fastPow(x0,1.f/b))); 

}

int main() {
  constexpr double Safety = 0.9;
  double eps=1.e-5;
  const float cut = std::pow(10.f/Safety,5.f);
  double accMin = eps/cut;
  std::cout << "eps/cut/accMin " << eps << " " << cut << " " << accMin
	    << std::endl;
  for (double acc=accMin; acc<eps;  acc*=10.)
    std::cout << acc << " " << std::pow(eps/acc,0.2) << " " << fastPow(eps/acc,0.2) << std::endl;

  for (float x=0.5f; x<20.f; x+=0.5f)
    std::cout << x << " " << std::pow(x,-1.0228f) << " " << fastPow(x,1.f-0.0228f)/x/x << ' ' << 1. -fastPow(x,1.f-0.0228f)/x/x/std::pow(x,-1.0228f)<< std::endl;
   
  float x = 3.5f;
  std::cout << x << " " << std::pow(x,4.5f) << " " << fastPow(x,4.5f) << ' ' << 1. -fastPow(x,4.5f)/std::pow(x,4.5f)<< std::endl;
  std::cout << x << " " << std::pow(x,4.5f) << " " << fastPow1(x,4.5f) << ' ' << 1. -fastPow1(x,4.5f)/std::pow(x,4.5f)<< std::endl;
  std::cout << x << " " << std::pow(std::pow(x,4.5f),1./4.5f) << " " << fastPow(fastPow(x,4.5f),1.f/4.5f) << std::endl;

  return 0;


}
