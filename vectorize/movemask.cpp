#include <x86intrin.h>

typedef float __attribute__( ( vector_size( 4*sizeof(float) ) ) ) V4F;
typedef double __attribute__( ( vector_size( 4*sizeof(double) ) ) ) V4D;
typedef double __attribute__( ( vector_size( 2*sizeof(double) ) ) ) V2D;


#ifdef __SSE2__
int mask(V4F m) {
  return _mm_movemask_ps(m);
}
int mask(V4D m) {
  V2D & l = (V2D&)(m[0]);
  V2D & h = (V2D&)(m[2]);
  return _mm_movemask_pd(l) | (_mm_movemask_pd(h)<<2);
}
#endif

#ifdef __AVX2__
int mask(V4D m) {
 return  _mm256_movemask_pd(m);
}
#endif


#include<iostream>
int main() {
{
  V4F a1 = V4F{-1,0,0,0};
  V4F a3 = V4F{0,0,1.,0} > 0;
  V4F a4 = V4F{0,0,0,1.} > 0;
  std::cout << a3[2] << ' ' << a4[2] << std::endl;
  std::cout << mask(a1) << ' ' <<mask(a3) << ' ' << mask(a4) << std::endl;
}

{
  V4D a1 = V4D{-1,0,0,0};
  V4D a3 = V4D{0,0,1.,0} > 0;
  V4D a4 = V4D{0,0,0,1.} > 0;
  std::cout << a3[2] << ' ' << a4[2] << std::endl;
  std::cout << mask(a1) << ' ' <<mask(a3) << ' ' << mask(a4) << std::endl;
}
 

 return 0;
}
