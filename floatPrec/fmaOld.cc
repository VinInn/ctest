#include<cmath>


float foo(float x, float y, float z) {
  return fmaf(x,y,z);
}


float bar(float x, float y, float z) {
  return x*y+z; 
}



inline float nofma(float x, float y, float z) __attribute__ ((__target__ ("no-fma"))); //  __attribute__ ((always_inline));
inline float nofma(float x, float y, float z)  {
  return (x*y)+z;
}


float poly(float const * a, float x, float b1, float b2) {
   float y = a[0] +x*(a[1] +x* a[2]);

   return nofma(b1,y,b2);

}

float xx[1024];
float yy[1024];
float loop(float const * a, float b1, float b2) {
  for (int i=0;i!=1024; ++i) yy[i]=poly(a,xx[i],b1,b2);
}
