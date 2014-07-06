namespace mathSSE {
  // 
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
}

#include<cmath>
#include<iostream>
#include<iomanip>
#include<typeinfo>

bool sng(double d) {
  return std::signbit(d);
}

bool sng(float d) {
  return std::signbit(d);
}

double cs(double a, double b) {
  return copysign(a,b);
}

float cs(float a, float b) {
  return copysignf(a,b);                                           
}


template<typename T>
void print(T a, T b) {
  using namespace mathSSE;
  std::cout << typeid(T).name() << " " 
            << a << " " << b << (samesign(a,b) ? " " : " not ") << "same sign" << std::endl;
}

int main() {
  using namespace mathSSE;
   // int mn = -0;
   // std::cout << mn << std::endl;
   // std::cout << std::hex << mn << std::endl;
  print(123,-902030);
  print(123LL,-902030LL);
  print(-123.f,123.e-4f);
  print(-123.,123.e-4);

  print(123, 902030);
  print(123LL,902030LL);
  print(123.f,123.e-4f);
  print(123.,123.e-4);

  print(-123,-902030);  
  print(-123LL,-902030LL);
  print(-123.f,-123.e-4f);
  print(-123.,-123.e-4);

  //  int const mask= 0x80000000;
  // std::cout << mask << std::endl;
  // std::cout << std::hex << mask << std::endl;
 
   return 0;
}
