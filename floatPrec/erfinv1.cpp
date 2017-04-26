#include <x86intrin.h>
#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cstring>
#include <array>

namespace approx_math {
  union binary32 {
    binary32() : ui32(0) {};
    binary32(float ff) : f(ff) {};
    binary32(int32_t ii) : i32(ii){}
    binary32(uint32_t ui) : ui32(ui){}
    
    uint32_t ui32; /* unsigned int */                
    int32_t i32; /* Signed int */                
    float f;
  };
}
/*  Quick and dirty, branchless, log implementations
    Author: Florent de Dinechin, Aric, ENS-Lyon 
    All right reserved
*/
template<int DEGREE>
inline float approx_logf_P(float p);
// degree =  5   => absolute accuracy is  16 bits
template<>
inline float approx_logf_P<5>(float y) {
  return  y * (float(0xf.ff652p-4) + y * (-float(0x8.0048ap-4) + y * (float(0x5.72782p-4) + y * (-float(0x4.20904p-4) + y * float(0x2.1d7fd8p-4))))) ;
}
// degree =  8   => absolute accuracy is  24 bits
template<>
inline float approx_logf_P<8>(float y) {
   return  y * ( float(0x1.00000cp0) + y * (float(-0x8.0003p-4) + y * (float(0x5.55087p-4) + y * ( float(-0x3.fedcep-4) + y * (float(0x3.3a1dap-4) + y * (float(-0x2.cb55fp-4) + y * (float(0x2.38831p-4) + y * (float(-0xf.e87cap-8) )))))))) ;
}
template<int DEGREE>
inline float unsafe_logf_impl(float x) {
  using namespace approx_math;

  binary32 xx,m;
  xx.f = x;
  
  // as many integer computations as possible, most are 1-cycle only, and lots of ILP.
  int e= (((xx.i32) >> 23) & 0xFF) -127; // extract exponent
  m.i32 = (xx.i32 & 0x007FFFFF) | 0x3F800000; // extract mantissa as an FP number
  
  int adjust = (xx.i32>>22)&1; // first bit of the mantissa, tells us if 1.m > 1.5
  m.i32 -= adjust << 23; // if so, divide 1.m by 2 (exact operation, no rounding)
  e += adjust;           // and update exponent so we still have x=2^E*y
  
  // now back to floating-point
  float y = m.f -1.0f; // Sterbenz-exact; cancels but we don't care about output relative error
  // all the computations so far were free of rounding errors...

  // the following is based on Sollya output
  float p = approx_logf_P<DEGREE>(y);
  

  constexpr float Log2=0xb.17218p-4; // 0.693147182464599609375 
  return float(e)*Log2+p;

}

template<int DEGREE>
inline float unsafe_logf(float x) {
  return unsafe_logf_impl<DEGREE>(x);
}


#define likely(x) (__builtin_expect(x, true))

inline 
float erfinv_like(float w) {
  w = w - 2.500000f;
  float p = 2.81022636e-08f;
  p = 3.43273939e-07f + p*w;
  p = -3.5233877e-06f + p*w;
  p = -4.39150654e-06f + p*w;
  p = 0.00021858087f + p*w;
  p = -0.00125372503f + p*w;
  p = -0.00417768164f + p*w;
  p = 0.246640727f + p*w;
  p = 1.50140941f + p*w;
  return p;
}

inline 
float erfinv_unlike(float w) {
  w = std::sqrt(w) - 3.000000f;
  float p = -0.000200214257f;
  p = 0.000100950558f + p*w;
  p = 0.00134934322f + p*w;
  p = -0.00367342844f + p*w;
  p = 0.00573950773f + p*w;
  p = -0.0076224613f + p*w;
  p = 0.00943887047f + p*w;
  p = 1.00167406f + p*w;
  p = 2.83297682f + p*w;
  return p;
}



inline float erfinv(float x) {
  float w, p;
  w = -  unsafe_logf<8>((1.0f-x)*(1.0f+x));
  // w = x*x;
  if ( w < 5.000000f )
    p =   erfinv_like(w);
  else 
    p = erfinv_unlike(w);
  return p*x;
}

/*
template<int N>
inline void erfinv(std::array<float,N> & r) {
  bool t[N];
  for (int i=0;i!=N;++i) {
    float w = -  unsafe_logf<8>((1.0f-r[i])*(1.0f+r[i]));
    t[i] = w > 5.000000f;
    b[i]=std::sqrt(2.f)*r[i]*erfinv_like(w);
  }
  
  for (int i=0;i!=NN;++i) {
    if(t[i]) {
      float w = -  unsafe_logf<8>((1.0f-r[i])*(1.0f+r[i]));
      b[i]=std::sqrt(2.f)*a[i]*erfinv_unlike(w);
    }
  }
}
*/


/*
inline float erfinv(float x) {
  float w, p;
  w = -  unsafe_logf<8>((1.0f-x)*(1.0f+x));
  if likely( w < 5.000000f ) {
      w = w - 2.500000f;
      p = 2.81022636e-08f;
      p = 3.43273939e-07f + p*w;
      p = -3.5233877e-06f + p*w;
      p = -4.39150654e-06f + p*w;
      p = 0.00021858087f + p*w;
      p = -0.00125372503f + p*w;
      p = -0.00417768164f + p*w;
      p = 0.246640727f + p*w;
      p = 1.50140941f + p*w;
    } else {
    w = std::sqrt(w) - 3.000000f;
    p = -0.000200214257f;
    p = 0.000100950558f + p*w;
    p = 0.00134934322f + p*w;
    p = -0.00367342844f + p*w;
    p = 0.00573950773f + p*w;
    p = -0.0076224613f + p*w;
    p = 0.00943887047f + p*w;
    p = 1.00167406f + p*w;
    p = 2.83297682f + p*w;
  }
  return p*x;
}

*/


constexpr int NN = 8*1024; 
float a[NN];
float b[NN];
typedef float __attribute__( ( vector_size( 32 ) ) ) float32x8_t;
typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;
float32x4_t va[NN/4];
float32x4_t vb[NN/4];


void compute() {
  for (int i=0;i!=NN;++i)
    b[i]=std::sqrt(2.f)*erfinv(a[i]);
}

void computeB() {
  bool t[NN];
  for (int i=0;i!=NN;++i) {
    float w = -  unsafe_logf<8>((1.0f-a[i])*(1.0f+a[i]));
    t[i] = w > 5.000000f;
    b[i]=std::sqrt(2.f)*a[i]*erfinv_like(w);
  }
  
  for (int i=0;i!=NN;++i) {
    if(t[i]) {
      float w = -  unsafe_logf<8>((1.0f-a[i])*(1.0f+a[i]));
      b[i]=std::sqrt(2.f)*a[i]*erfinv_unlike(w);
    }
  }
}

void computeB1() {
  int t[NN]; int k=0;
  float w[NN];
  for (int i=0;i!=NN;++i) {
    w[i] = -  unsafe_logf<8>((1.0f-a[i])*(1.0f+a[i]));
    b[i]=std::sqrt(2.f)*a[i]*erfinv_like(w[i]);
  }
   for (int i=0;i!=NN;++i) if (w[i]>5.f) t[k++] = i;

  for (int j=0;j!=k;++j) {
    auto i = t[j];
    float w = -  unsafe_logf<8>((1.0f-a[i])*(1.0f+a[i]));
    b[i]=std::sqrt(2.f)*a[i]*erfinv_unlike(w);
  }
}

void computeB2() {
  float w[NN];
  for (int i=0;i!=NN;++i) {
    w[i] = -  unsafe_logf<8>((1.0f-a[i])*(1.0f+a[i]));
    b[i]=std::sqrt(2.f)*a[i]*erfinv_like(w[i]);
  }
  for (int i=0;i!=NN;++i) {
    if(w[i]>5.f) {
      b[i]=std::sqrt(2.f)*a[i]*erfinv_unlike(w[i]);
    }
  }
}


template<typename Vec, typename F> 
inline
Vec apply(Vec v, F f) {
  typedef typename std::remove_reference<decltype(v[0])>::type T;
  constexpr int N = sizeof(Vec)/sizeof(T);
  Vec ret;
  for (int i=0;i!=N;++i) ret[i] = f(v[i]);
  return ret;
}


  template<typename V>
  bool testz(V const t) {
   return _mm_testz_si128(__m128i(t),__m128i(t));
  }

inline
void computeOne(int i) {
  float32x4_t v = (1.0f-va[i])*(1.0f+va[i]);
  float32x4_t w = - apply(v,unsafe_logf<8>);
  auto t = w > 5.000000f;
  //    if likely(_mm256_movemask_ps(mask) == 255) {
  if (testz(t)) {
    vb[i]=std::sqrt(2.f)*va[i]*apply(w,erfinv_like);
  } else {
    vb[i]=std::sqrt(2.f)*apply(va[i],erfinv);
  }
}

bool foo[NN/4];
void computeV() {
  for (int i=0;i!=NN/4;++i) 
    if likely(foo[i]) computeOne(i);
    else vb[i] = apply(va[i],sqrtf);
}

int count() {
  int n=0;
 for (int i=0;i!=NN;++i)
   if (-unsafe_logf<8>((1.0f-a[i])*(1.0f+a[i])) < 5.f) n++;
 return n;
}


#include<random>
#include<iostream>
std::mt19937 eng;
std::mt19937 eng2;
std::uniform_real_distribution<float> rgen(0.,1.);
 

void fill() {
  for (int i=0;i!=NN;++i)
    a[i]=2.*rgen(eng)-1.;
}

float sum() {
  float q=0;
  for (int i=0;i!=NN;++i) q+=b[i];
  return q;
}

unsigned int taux=0;
inline volatile unsigned long long rdtsc() {
 return __rdtscp(&taux);
}


int main(int argc, char**) {

  if (argc<4)
  for (int i=0;i!=NN/4;++i) foo[i]=true;
 
  long long t1=0;
  float s1=0;
  long long t2=0;
  float s2=0;
  long long t3=0;
  float s3=0;
  

  fill();
  compute();
  std::cout << count() << " / " << NN << std::endl;

  /*
  for (int i=0;i!=NN;++i) std::cout << a[i] << " ";
  std::cout <<	std::endl;		     
  for (int i=0;i!=NN;++i) std::cout << b[i] << " ";
  std::cout <<	std::endl;		     
  */

  for (int i=0; i!=10000; ++i) {
    fill();
    t1 -= rdtsc();
    compute();
    t1 += rdtsc();
    s1+=sum();

    t2 -= rdtsc();
    computeB();
    t2 += rdtsc();
    s2+=sum();

    memcpy(va,a,NN*4);
    t3 -= rdtsc();
    computeV();
    t3 += rdtsc();
    memcpy(b,vb,NN*4);
    s3+=sum();


  }
  std::cout << s1 << " " << double(t1)/10000 << std::endl;
  std::cout << s2 << " " << double(t2)/10000 << std::endl;
  std::cout << s3 << " " << double(t3)/10000 << std::endl;
}
