#include<cmath>
inline
float foo_like(float w) {
  w = w - 2.500000f;
  return w;
}

inline 
float foo_unlike(float w) {
  w = std::sqrt(w) - 3.000000f;
  return w;
}

inline float foo(float x) {
  float w, p;
  w = x*x;
  if ( w < 5.000000f )
    p =   foo_like(w);
  else 
    p = foo_unlike(w);
  return p*x;
}

constexpr int NN = 8*1024; 
float a[NN];
float b[NN];


void compute() {
  for (int i=0;i!=NN;++i)
    b[i]=foo(a[i]);
}
