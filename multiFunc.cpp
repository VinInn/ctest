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

