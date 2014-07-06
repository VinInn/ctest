 
#include<cmath>

template<typename T> inline bool samesign(T rh, T lh);

  template<>
  inline bool
  __attribute__((always_inline)) __attribute__ ((pure)) samesign<int>(int rh, int lh) {
    int const mask= 0x80000000;
    return ((rh^lh)&mask) == 0;
  }

  template<>
  inline bool
  __attribute__((always_inline)) __attribute__ ((pure)) samesign<long long>(long long rh, long long lh) {
    long long const mask= 0x8000000000000000LL;
    return ((rh^lh)&mask) == 0;
  }

  template<>
  inline bool
  __attribute__((always_inline)) __attribute__ ((pure)) samesign<float>(float rh, float lh) {
    union { int i; float f; } a, b;
    a.f=rh; b.f=lh;
    return samesign<int>(a.i,b.i);
  }

  template<>
  inline bool
  __attribute__((always_inline)) __attribute__ ((pure)) samesign<double>(double rh, double lh) {
    union { long long i; double f; } a, b;
    a.f=rh; b.f=lh;
    return samesign<long long>(a.i,b.i);
  }


bool ss1(float a, float b) {
  return samesign(a,b);
}

bool ss2(float a, float b) {
  return std::signbit(a)==std::signbit(b);
}

bool ss3(float a, float b) {
  return a*b>0;
}


float cs1(bool s, float x) {
  return s ? -x : x;
}

struct ChangeSignF {
  ChangeSignF(bool s) : cs(s ? 0x80000000 : 0x0){}
  float operator()(float x) const{
    union { float f; unsigned int i; } xx; 
    xx.f=x; xx.i^=cs;
    return xx.f;
  }  

  unsigned int cs;
};

float cs2(ChangeSignF s, float x) {
   return s(x);

}

#include<iostream>
int main() {

  std::cout << cs1(true,-1)  << std::endl;;
  std::cout << cs1(false,-1)  << std::endl;;

  std::cout << cs2(true,-1)  << std::endl;;
  std::cout << cs2(false,-1)  << std::endl;;

  return 0;
}
