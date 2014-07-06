typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;
typedef float __attribute__( ( vector_size( 32 ) ) ) float32x8_t;


template<typename Float>
inline
Float atan(Float t) {
  constexpr float PIO4F = 0.7853981633974483096f;

  Float z= (t > 0.4142135623730950f) ? (t-1.0f)/(t+1.0f) : t;
  Float ret; ret = ( t > 0.4142135623730950f ) ? PIO4F : ret;
  Float z2 = z * z;
  ret +=
    ((( 8.05374449538e-2f * z2
	- 1.38776856032E-1f) * z2
      + 1.99777106478E-1f) * z2
     - 3.33329491539E-1f) * z2 * z
    + z;
  
  // move back in place
  //  return ( t > 0.4142135623730950f ) ? ret+PIO4F : ret;
  return ret;
}

float32x8_t va[1024];
float32x8_t vb[1024];

float a[8*1024];
float b[8*1024];

void computeV() {
  for (int i=0;i!=1024;++i)
    vb[i]=atan(va[i]);
}

//inline
void computeL() {
  for (int i=0;i!=8*1024;++i)
    b[i]=atan(a[i]);
}
