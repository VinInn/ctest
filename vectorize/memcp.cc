#include <string.h>

typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;
typedef double __attribute__( ( vector_size( 32 ) ) ) float64x4_t;
typedef double __attribute__( ( vector_size( 64 ) ) ) float64x8_t;
typedef int __attribute__( ( vector_size( 16 ) ) )  int32x4_t;


template<typename F, typename I>
inline I toI(F f) { I i; memcpy(&i,&f, sizeof(F)); return i;}
template<typename F, typename I>   
inline F toF(I i) { F f; memcpy(&f,&i, sizeof(F)); return f;}



float32x4_t min(float32x4_t a, float32x4_t b) {
  int32x4_t m = a < b;
  return toF<float32x4_t,int32x4_t>(
      (toI<float32x4_t,int32x4_t>(a)&m) |
      (toI<float32x4_t,int32x4_t>(b)&(~m))
      );

}

float32x4_t min2(float32x4_t a, float32x4_t b) {
  return (a<b) ? a : b;
}
