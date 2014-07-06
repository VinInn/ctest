#include<cmath>
#include<algorithm>
inline
void sincos(float&s,float&c,float x) {
   s=x*(1.f+3.f*x*x);
   c=std::exp(x*x);
   if (x<0) std::swap(s,c);   
}

inline float icos(float x) {
float s; float c; sincos(s,c,x);
return c;
}

inline float isin(float x) {
float s; float c; sincos(s,c,x); 
return s;
}


void rot(float&x, float&y, float a) {
  float u = x*icos(a)+y*isin(a);
  float v = x*isin(a)-y*icos(a);
  x=u; y=v;
}
