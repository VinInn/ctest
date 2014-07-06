namespace {
inline
float _sum0(float const *  x, 
           float const *  y, float const *  z) {
  float sum=0;
  for (int i=0; i!=1024; ++i)
    sum += z[i]+x[i]*y[i];
  return sum;
}
}


float  __attribute__ ((__target__ ("arch=haswell")))
sum1(float const *  x,
     float const *  y, float const *  z) {
  return _sum0(x,y,z);
}

float  __attribute__ ((__target__ ("arch=nehalem")))
sum1(float const *  x,
     float const *  y, float const *  z) {
  return _sum0(x,y,z);
}


float  __attribute__ ( (__target__("arch=nehalem"), __target__("arch=haswell")) )
sum0(float const *  x, 
      float const *  y, float const *  z) {
 float sum=0;
 for (int i=0; i!=1024; ++i)
   sum += z[i]+x[i]*y[i];
 return sum;
}

