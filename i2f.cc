
#include<cmath>

constexpr float a = pow(2,-149);
constexpr float b = std::pow(2,-149);

constexpr float c = floorf(-3.12f);
constexpr float d = std::floor(-3.12f);


constexpr float e = cosf(3.12f);
constexpr float f = std::cos(3.12f);


union fi {
  constexpr fi(int x) : i(x){} 
  constexpr fi(float x) : f(x){} 
  float f; int i; 
};

inline constexpr float i2f(int x) {
  return fi(x).f;
}


inline constexpr int f2i(float x) {
  return fi(x).i;
}


float __attribute__ ((aligned(16))) f[1024];
int __attribute__ ((aligned(16))) i[1024];


void t1() {
  for (int j=0; j!=1024; ++j)
    f[j] = i2f(i[j]);
}


void t2() {
  for (int j=0; j!=1024; ++j)
    i[j] = f2i(f[j]);
}


void tt() {
  for (int j=0; j!=1024; ++j) {
    int n = f2i(f[j]); 
    // fractional part
    constexpr int inv_mant_mask =  ~0x7f800000;
    constexpr int p05 = f2i(0.5f);
    n &= inv_mant_mask;
    n |= p05;
    f[j] = i2f(n);
  }
}
