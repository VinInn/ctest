// #define DIV
using Float = float;
// using Float = double;

namespace {
inline
Float _sum0(Float const *  x, 
           Float const *  y, Float const *  z, int n) {
  Float sum=0;
#pragma GCC ivdep
  for (int i=0; i!=n; ++i)
#ifdef DIV
    sum += z[i]+x[i]/y[i];
#else
    sum += z[i]+x[i]*y[i];
#endif
  return sum;
}
}

Float __attribute__ ((target ("default")))
sum(Float const *  x,
     Float const *  y, Float const *  z, int n) {
  return _sum0(x,y,z,n);
}


Float  __attribute__ ((__target__ ("arch=haswell")))
sum(Float const *  x,
     Float const *  y, Float const *  z, int n) {
  return _sum0(x,y,z,n);
}

Float  __attribute__ ((__target__ ("arch=sandybridge")))
sum(Float const *  x,
     Float const *  y, Float const *  z, int n) {
  return _sum0(x,y,z,n);
}



Float  __attribute__ ((__target__ ("arch=nehalem")))
sum(Float const *  x,
     Float const *  y, Float const *  z, int n) {
  return _sum0(x,y,z,n);
}


#include <x86intrin.h>
unsigned int taux=0;
inline unsigned long long rdtscp() {
 return __rdtscp(&taux);
}

#include<cstdlib>
#include <fstream>
#include <iostream>

int main(int npar, char * par[]) {

  alignas(32) Float x[10240],y[10240],z[10240];
  for (int i=0; i!=10240; ++i)
    x[i]=y[i]=z[i]=(1+i)*0.1;

  
  int n = (npar>1) ? atoi(par[1]) : 1024;
  int n2 = (npar>2) ? atoi(par[2]) : 1024;
  int n3 = (npar>3) ? atoi(par[3]) : 1024;


  long long t = -rdtscp();
  Float s = _sum0(x,y,z,1024);  
  t += rdtscp();
  std::cout << "generic 1024 " << double(t)/1024 << ' ' << s << std::endl;

  t = -rdtscp();
  s = sum(x,y,z,n);
  t += rdtscp();
  std::cout << "native " << n << ' '<< double(t)/std::max(n,1) << ' ' << s << std::endl;

  t = -rdtscp();
  s = sum(x,y,z,n2);
  t += rdtscp();
  std::cout << "native "<< n2 << ' ' << double(t)/std::max(n2,1) << ' ' << s << std::endl;

  t = -rdtscp();
  s = _sum0(x,y,z,n3);
  t += rdtscp();
  std::cout << "generic " << n3 << ' '<< double(t)/std::max(n3,1) << ' ' << s << std::endl;


  t = -rdtscp();
  for (int i=0; i<100; ++i) 
    s += sum(x,y,z,1024);
  t += rdtscp();
  std::cout << "native 100*1024 " << double(t)/(100*1024) << ' ' << s << std::endl;

 t = -rdtscp();
  for (int i=0; i<1000; ++i)
    s += sum(x,y,z,1024);
  t += rdtscp();
  std::cout << "native 1000*1024 " << double(t)/(1000*1024) << ' ' << s << std::endl;

 t = -rdtscp();
  for (int i=0; i<100; ++i)
    s += sum(x,y,z,10240);
  t += rdtscp();
  std::cout << "native 100*10240 " << double(t)/(1000*1024) << ' ' << s << std::endl;

 t = -rdtscp();
  for (int i=0; i<100; ++i)
    s += _sum0(x,y,z,10240);
  t += rdtscp();
  std::cout << "generic 100*10240 " << double(t)/(1000*1024) << ' ' << s << std::endl;

  t = -rdtscp();
  for (int i=0; i<100; ++i)
    s += sum(x,y,z,1024);
  t += rdtscp(); 
  std::cout << "native 100*1024 " << double(t)/(100*1024) << ' ' << s << std::endl;


  while (1) {
    {
    std::ifstream in("endAvx");
    if (in) return 0;
    in.close();
    }
    std::ifstream in("goAvx");
    if(in) {
     for (int i=0; i<1000; ++i)
      s += sum(x,y,z,10240);
    } else {
     for (int i=0; i<1000; ++i)
      s += _sum0(x,y,z,10240);
    }
    in.close();
  }

  return 0;
}
