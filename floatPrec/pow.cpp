#include<iostream>
#include<cmath>
#include<cstring>


inline
float itof(int i) { float f;  memcpy(&f,&i,sizeof(float)); return f;}
inline
int ftoi(float f) { int i;  memcpy(&i,&f,sizeof(float)); return i;}

inline 
float ilog(float x) { return ((ftoi(x)>> 23) & 0xFF) -127; }

inline
float iexp(float x) { 
  // constexpr float inv_log2f = float(0x1.715476p0);
  float z = std::round(x);
  // constexpr float log2F = 0xb.17218p-4;
  int e = z;
  return itof((e+127)<<23);
}

inline
float nrPow(float y0, float x, float a) {
  return  y0*(1.f-a*(1.f-x*iexp(-ilog(y0)/a))); 
}


int main() {
  std::cout << itof(128<<23) << std::endl;
  std::cout << ftoi(2.f) << std::endl;
  std::cout << ilog(4.f) << std::endl;
  std::cout << ilog(0.25f) << std::endl;
  std::cout << iexp(3.f) << std::endl;
  std::cout << iexp(-1.f) << std::endl;

   float a = 0.577;

   float x = 1./1024.;

  std::cout << ilog(x) << std::endl;

  std::cout << "pow " << std::pow(x,a) << std::endl;
  std::cout << "pow " << std::exp(a*std::log(x)) << std::endl;
  std::cout << "oOr " << iexp(a*ilog(x)) << std::endl;
  std::cout << "o1r " << nrPow(iexp(a*ilog(x)),x,a) << std::endl;
   


   return 0;

}
