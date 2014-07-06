typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;
typedef double  __attribute__( ( vector_size( 32 ) ) ) float64x4_t;


inline 
float64x4_t convert(float32x4_t f) {
  return float64x4_t{f[0],f[1],f[2],f[3]};
}


inline
float dotf(float32x4_t x, float32x4_t y) {
  float ret=0;
  for (int i=0;i!=4;++i) ret+=x[i]*y[i];
  return ret;
}


inline
double dotd(float64x4_t x, float64x4_t y) {
  double ret=0;
  for (int i=0;i!=4;++i) ret+=x[i]*y[i];
  return ret;
}


float ddot(float32x4_t x, float32x4_t y,  float32x4_t z) {
  return dotf(x,y)*dotf(y,z);
}
double ddot(float64x4_t x, float64x4_t y,  float64x4_t z) {
  return dotd(x,y)*dotd(y,z);
}
// double ddot(float64x4_t x, float64x4_t y,  float64x4_t z) {
float ddotd(float32x4_t x, float32x4_t y,  float32x4_t z) {
  float64x4_t dx=convert(x);
  float64x4_t dy=convert(y);
  float64x4_t dz=convert(z);
  return dotd(dx,dy)*dotd(dy,dz);
}



float dotd1(float32x4_t x, float32x4_t y) {
  float64x4_t dx,dy;
  for (int i=0;i!=4;++i) {
    dx[i]=x[i]; dy[i]=y[i];
  }
  double ret=0;
  for (int i=0;i!=4;++i) ret+=dx[i]*dy[i];
  return ret;
}

float dotd2(float32x4_t x, float32x4_t y) {
  float64x4_t dx=convert(x);
  float64x4_t dy=convert(y);
  return dotd(dx,dy);
}


float dotd21(float32x4_t x, float32x4_t y) {
  auto dotL = [](float64x4_t x, float64x4_t y) {
  double ret=0;
  for (int i=0;i!=4;++i) ret+=x[i]*y[i];
  return ret;
    };
  float64x4_t dx=convert(x);
  float64x4_t dy=convert(y);
  return dotL(dx,dy);
}



float dotd3(float32x4_t x, float32x4_t y) {
  float64x4_t dx{x[0],x[1],x[2],x[3]};
  float64x4_t dy{y[0],y[1],y[2],y[3]};
  double ret=0;
  for (int i=0;i!=4;++i) ret+=dx[i]*dy[i];
  return ret;
}

float dotd4(float32x4_t x, float32x4_t y) {
  float64x4_t dx,dy;
  for (int i=0;i!=4;++i) {
    dx[i]=x[i]; dy[i]=y[i];
  }
  return dotd(dx,dy);
}
