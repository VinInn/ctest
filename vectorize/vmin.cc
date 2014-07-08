typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;

  template<typename V1>
  V1 vmax(V1 a, V1 b) {
    return (a>b) ? a : b;
  }
  template<typename V1>
  V1 vmin(V1 a, V1 b) {
    return (a<b) ? a : b;
  }
 

float foo(float a, float b, float c) {
  return vmin(vmax(a,b),c);
}

float32x4_t foo(float32x4_t a, float32x4_t b, float32x4_t c) {
  return vmin(vmax(a,b),c);
}

template<typename Float>
Float bart(Float a) { 
  constexpr Float zero{0.f};
  constexpr Float it = zero+float(0x5.8b90cp4);
  // log of the smallest normal
  constexpr Float zt = zero-float(0x5.75628p4);
  return vmin(vmax(a,zt),it);
}


float bar(float a) {
   return bart(a);
}
float32x4_t bar(float32x4_t a) {
   return bart(a);
}
