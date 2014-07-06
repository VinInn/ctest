namespace {
inline
float _sum0(float const *  x, 
           float const *  y, float const *  z, int n) {
  float sum=0;
#pragma GCC ivdep
  for (int i=0; i!=n; ++i)
    sum += z[i]+x[i]*y[i];
  return sum;
}
}

float __attribute__ ((target ("default")))
sum(float const *  x,
     float const *  y, float const *  z, int n) {
  return _sum0(x,y,z,n);
}


float  __attribute__ ((__target__ ("arch=haswell")))
sum(float const *  x,
     float const *  y, float const *  z, int n) {
  return _sum0(x,y,z,n);
}

float  __attribute__ ((__target__ ("arch=nehalem")))
sum(float const *  x,
     float const *  y, float const *  z, int n) {
  return _sum0(x,y,z,n);
}


#include <x86intrin.h>
unsigned int taux=0;
inline unsigned long long rdtscp() {
 return __rdtscp(&taux);
}

#include<cstdlib>
#include<iostream>
int main(int npar, char * par[]) {

  float x[1024],y[1024],z[1024];
  for (int i=0; i!=1024; ++i)
    x[i]=y[i]=z[i]=i;

  
  int n = (npar>1) ? atoi(par[1]) : 1024;

  long long t = -rdtscp();
  float s = _sum0(x,y,z,1024);  
  t += rdtscp();
  std::cout << "generic " << t << ' ' << s << std::endl;

  t = -rdtscp();
  s = sum(x,y,z,n);
  t += rdtscp();
  std::cout << "native 0 " << t << ' ' << s << std::endl;


  t = -rdtscp();
  for (int i=0; i<100; ++i) 
    s += sum(x,y,z,1024);
  t += rdtscp();
  std::cout << "native 100 " << t << ' ' << s << std::endl;



  return 0;
}
