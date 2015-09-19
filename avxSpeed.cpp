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
sum(Float const *  x, Float const *  y, Float const *  z, int n) {
  return _sum0(x,y,z,n);
}


Float  __attribute__ ((__target__ ("skylake")))
sum(Float const *  x, Float const *  y, Float const *  z, int n) {
  return _sum0(x,y,z,n);
}


Float  __attribute__ ((__target__ ("arch=haswell")))
sum(Float const *  x, Float const *  y, Float const *  z, int n) {
  return _sum0(x,y,z,n);
}

Float  __attribute__ ((__target__ ("arch=sandybridge")))
sum(Float const *  x,
     Float const *  y, Float const *  z, int n) {
  return _sum0(x,y,z,n);
}



Float  __attribute__ ((__target__ ("arch=nehalem")))
sum(Float const *  x, Float const *  y, Float const *  z, int n) {
  return _sum0(x,y,z,n);
}

#include<cstdlib>
#include <fstream>
#include <iostream>

int main(int npar, char * par[]) {
  Float s=0;
  alignas(32) Float x[10240],y[10240],z[10240];
  for (int i=0; i!=10240; ++i)
    x[i]=y[i]=z[i]=(1+i)*0.1;

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

