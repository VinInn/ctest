#include <x86intrin.h>

typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;

inline
float32x4_t atan(float32x4_t t) {
  constexpr float PIO4F = 0.7853981633974483096f;
  float32x4_t high = t > 0.4142135623730950f;
  auto z = t;
  float32x4_t ret={0.f,0.f,0.f,0.f};
    // if all low no need to blend
  if ( _mm_movemask_ps(high) != 0) {
    z   = ( t > 0.4142135623730950f ) ? (t-1.0f)/(t+1.0f) : t;
    ret = ( t > 0.4142135623730950f ) ? ret+PIO4F : ret;
  }
  /*
  auto z2 = z * z;
  ret +=
    ((( 8.05374449538e-2f * z2
	- 1.38776856032E-1f) * z2
      + 1.99777106478E-1f) * z2
     - 3.33329491539E-1f) * z2 * z
    + z;
  */
  return  ret += z;
}



float32x4_t doAtan(float32x4_t z) { return atan(z);}

float32x4_t va[1024];
float32x4_t vb[1024];

void computeV() {
  for (int i=0;i!=1024;++i)
    vb[i]=atan(va[i]);
}

