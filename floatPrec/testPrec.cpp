// clang++ -fsanitize=numerical -O2 -Wall -g ~innocent/public/ctest/floatPrec/testPrec.cpp -DFLOAT=double
#include<cstdio>

#ifndef FLOAT
#error please define FLOAT
#endif

using Float = FLOAT;

int main() {

  const Float t = 1.;
  Float r = 1;
  Float a = 1.;
  const Float h = 0.5;
  while (a==r)  {
    a*=h;
    r = (t + a) - t;
  } 
  printf("%f %a\n",a,a);
  printf("%f %a\n",r,r);
  r = (__float128(t) + __float128(a)) - __float128(t);
  printf("%f %a\n",r,r);
  return a;
}
