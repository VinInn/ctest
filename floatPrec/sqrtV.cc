#include<cmath>
#include<type_traits>

typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;
typedef float __attribute__( ( vector_size( 32 ) ) ) float32x8_t;
typedef int __attribute__( ( vector_size( 32 ) ) ) int32x8_t;

float32x8_t va[1024];
float32x8_t vb[1024];
float32x8_t vc[1024];

template<typename Vec, typename F> 
inline
Vec apply(Vec v, F f) {
  typedef typename std::remove_reference<decltype(v[0])>::type T;
  constexpr int N = sizeof(Vec)/sizeof(T);
  Vec ret;
  for (int i=0;i!=N;++i) ret[i] = f(v[i]);
  return ret;
}

void computeOne() {
    vb[0]=apply(va[0],sqrtf);
}

void computeS() {
  for (int i=0;i!=1024;++i)
    vb[i]=apply(va[i],sqrtf);
}

void computeL() {
  for (int i=0;i!=1024;++i)
    for (int j=0;j!=8;++j)
    vb[i][j]=sqrtf(va[i][j]);
}

template<typename Float>
inline
Float atanF(Float t) {
  constexpr float PIO4F = 0.7853981633974483096f;
  Float z= (t > 0.4142135623730950f) ? (t-1.0f)/(t+1.0f) : t;
  // if( t > 0.4142135623730950f ) // * tan pi/8 
  
  Float z2 = z * z;
  Float ret =
    ((( 8.05374449538e-2f * z2
	- 1.38776856032E-1f) * z2
      + 1.99777106478E-1f) * z2
     - 3.33329491539E-1f) * z2 * z
    + z;
  
  // move back in place
  return ( t > 0.4142135623730950f ) ? ret : ret + PIO4F;
  return ret;
}

void computeA() {
  for (int i=0;i!=1024;++i)
    vb[i]=apply(va[i],atanF<float>);
}

void computeAL() {
  for (int i=0;i!=1024;++i)
    for (int j=0;j!=8;++j)
      vb[i][j]=atanF(va[i][j]);
}
