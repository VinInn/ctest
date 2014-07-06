
inline float poly(float x) {
  static constexpr float a1[3]{-1.f,2.f,0.45f};
  static constexpr float a2[3]{1.5f,3.2f,-0.45f};
  static constexpr float a3[3]{1.f,-2.f,0.55f};
  static constexpr float a4[3]{-1.5f,3.2f,-0.55f};

  //float  a[3];
  //for (int i=0;i<3; ++i) a[i] = (x>0) ? ( (x >2.f) ? a1[i] : a2[i] ) : ( (x >-2.f) ? a3[i] : a4[i] );
  //  auto a =  (x>0) ? ( (x >2.f) ? a1 : a2 ) : ( (x >-2.f) ? a3 : a4 );
  //return a[0] + x*(a[1]+x*a[2]);

  auto a = [&](int i) { return (x>0) ? ( (x >2.f) ? a1[i] : a2[i] ) : ( (x >-2.f) ? a3[i] : a4[i] );};
  return a(0) + x*(a(1)+x*a(2));

  
}

inline float poly2(float x) {
  static constexpr float a1[3]{-1.f,2.f,0.45f};
  static constexpr float a2[3]{1.5f,3.2f,-0.45f};
  // static constexpr float a3[3]{1.f,-2.f,0.55f};
  // static constexpr float a4[3]{-1.5f,3.2f,-0.55f};

  return (x>0) ? a1[0] + x*(a1[1]+x*a1[2]) :  a2[0] + x*(a2[1]+x*a2[2]);

}

inline float poly3(float x) {
  static constexpr float a[4]{-1.f,2.f,0.45f, -1.3};
  static constexpr float b[4]{1.5f,3.2f,-0.45f, 1.3f};
  static constexpr float c[4]{1.f,-2.f,0.55f, 0.2f};

  int i = (x>0) ? ( (x >2.f) ? 0 : 1 ) : ( (x >-2.f) ? 2 : 3 );

  return  a[i] + x*(b[i]+x*c[i]);

}




float x[1024], y[1024];

void compute() {
  for (int i=0; i<1024; ++i) y[i] = poly(x[i]);
}

void compute2() {
  for (int i=0; i<1024; ++i) y[i] = poly2(x[i]);
}

void compute3() {
  for (int i=0; i<1024; ++i) y[i] = poly3(x[i]);
}


typedef float __attribute__( ( vector_size( 32 ) ) ) float32x8_t;

float32x8_t poly(float32x8_t x ) {
  static constexpr float a1[3]{-1.f,2.f,0.45f};
  static constexpr float a2[3]{1.5f,3.2f,-0.45f};
  static constexpr float a3[3]{1.f,-2.f,0.55f};
  static constexpr float a4[3]{-1.5f,3.2f,-0.55f};
  float32x8_t  a[3];
  for (int i=0;i<3; ++i) a[i] = (x>0) ? ( (x >2.f) ? a1[i] : a2[i] ) : ( (x >-2.f) ? a3[i] : a4[i] );
  return a[0] + x*(a[1]+x*a[2]);
}
