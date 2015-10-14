
template<int DEGREE>
inline float approx_expf_P(float p);

// degree =  6   => absolute accuracy is  27 bits
template<>
inline float approx_expf_P<6>(float y) {
#if HORNER  // HORNER 
  float p =  float(0x2.p0) + y * (float(0x2.p0) + y * (float(0x1.p0) + y * (float(0x5.55523p-4) + y * (float(0x1.5554dcp-4) + y * (float(0x4.48f41p-8) + y * float(0xb.6ad4p-12)))))) ;
#else // ESTRIN does seem to save a cycle or two
  float p56 = float(0x4.48f41p-8) + y * float(0xb.6ad4p-12);
  float p34 = float(0x5.55523p-4) + y * float(0x1.5554dcp-4);
  float y2 = y*y;
  float p12 = float(0x2.p0) + y; // By chance we save one operation here! Funny.
  float p36 = p34 + y2*p56;
  float p16 = p12 + y2*p36;
  float p =  float(0x2.p0) + y*p16;
#endif
  return p;
}


  void comp(float * b, float const * a, int N) {
    for (int i=0; i<N; ++i) {
      b[i] = approx_expf_P<6>(a[i]);
    }
  }

void init(float * x, int N, float y) {
   for ( int i = 0; i < N; ++i ) x[i]=y;
}


float * alloc(int N) {
  return new float[N];

}


int main() {

   int N = 1000;

   int size = N*N;
   float * a = alloc(size);
   float * b = alloc(size);
   float * c = alloc(size);

  init(c,size,0.f);
  init(a,size,1.3458f);
  init(b,size,2.467f);

  comp(c,a,size);

}
