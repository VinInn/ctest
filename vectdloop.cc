#include<cmath>

inline float fn(float x) {
  return 2.f*x+std::sqrt(x);
}


void  __attribute__((noinline)) sloop(float * restrict s, float const * restrict xx) {
  // asm ("");
  const int ls=16;
  for (int j=0; j < ls; ++j) {
    s[j] = fn(xx[j]);
  } 
}

int dloop(float yyy) {
  int niter = 100000;
  float x = 0.5f; yyy=0;
  const int ls=16;
  for (int i=0; i < niter; ++i) { 
    float s[ls]; float xx[ls];
    for (int j=0; j < ls; ++j) xx[j] =x+(5*(j&1));
    sloop(s,xx);
    // for (int j=0; j < ls; ++j)  s[j] = fn(xx[j]); 
    x += 1e-6f;
    for (int j=0; j < ls; ++j) yyy+=s[j];
  }
  if (yyy == 2.32132323232f) niter--; 
  return niter;
}
