typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;
typedef double __attribute__( ( vector_size( 32 ) ) ) float64x4_t;
typedef double __attribute__( ( vector_size( 64 ) ) ) float64x8_t;

typedef int __attribute__( ( vector_size( 16 ) ) ) int32x4_t;


typedef float __attribute__( ( vector_size( 16 ) , aligned(4) ) ) float32x4a4_t;


template<typename T, int N>
struct ExtVecTraits {
  typedef T __attribute__( ( vector_size( N*sizeof(T) ) ) ) type;
  typedef T __attribute__( ( vector_size( N*sizeof(T) ), aligned(alignof(T)) ) ) typeA;

  static type load(T const * p) { return *(typeA const *)(p);}
  static void load(T *p, type const & v) { *(typeA *)(p) = v; }

  static typeA & bind(T * p) { return *(typeA *)(p);}
  static typeA & bind(T const * p) { return *(typeA const *)(p);}


};

template<typename T, int N> using ExtVec =  typename ExtVecTraits<T,N>::type;

template<typename T> using Vec4D = ExtVec<T,4>;


#include<cmath>

using V4 = float32x4_t;

// constexpr float a[4] = {[0 ... 3]=3.14f};
//constexpr V4 va[4] = {[0 ... 3]=3.14f};

V4  ebuild(float x) { return x+V4{}; }


constexpr V4  build(float x) { return x+V4{0,0,0,0}; }

constexpr V4 a = build(3.12f);

float64x4_t convert(float32x4_t x) {
   return float64x4_t{x[0],x[1],x[2],x[3]};
}

float32x4_t convert(float64x4_t x) {
   return float32x4_t{x[0],x[1],x[2],x[3]};
}


float32x4_t loadIt(float const * x) {
   return ExtVecTraits<float,4>::load(x);
}

float32x4a4_t loadA4(float const * x) {
   return *(float32x4a4_t const *)(x);
}

float32x4_t loadV4(float32x4a4_t x) {
   return float32x4_t{x[0],x[1],x[2],x[3]};
}


float64x4_t loadIt(double const * x) {
   return ExtVecTraits<double,4>::load(x);
}


float32x4_t shuffleA(float32x4_t x) {
   return float32x4_t{x[0],x[0],x[1],x[1]};
}

float32x4_t shuffleB(float32x4_t x) {
   return __builtin_shuffle(x,int32x4_t{0,0,1,1});
}

float32x4_t shuffle2A(float32x4_t const & x) {
   return float32x4_t{x[1],x[0],x[3],x[2]};
}

float32x4_t shuffle2B(float32x4_t const & x) {
   return __builtin_shuffle(x,int32x4_t{1,0,3,2});
}




float  sum(float32x4_t x) {
  return x[0]+x[1]+x[2]+x[3];
}

float  suml(float32x4_t x) {
  float r=0;
  for (int i=0;i<4; ++i) r+=x[i];
  return r;
}



float  sum(float32x4_t x, float32x4_t y) {
  return sum(x*y);
}

float32x4_t  prod1(float32x4_t x, float32x4_t y) {
  return x[1]*y;
}

float32x4_t  prodM(float32x4_t x, float32x4_t y[3]) {
  return x[0]*y[0]+x[1]*y[1]+x[2]*y[2];
}


float32x4_t  div(float32x4_t x, float32x4_t y) {
  return x/y;
}


float32x4_t sqrt(float32x4_t x) {
  return float32x4_t{std::sqrt(x[0]),std::sqrt(x[1]),std::sqrt(x[2]),std::sqrt(x[3])};
}

float32x4_t sqrtl(float32x4_t x) {
  float32x4_t r;
  for (int i=0;i!=4; ++i) r[i]=std::sqrt(x[i]);
  return r;
}


float64x4_t dfmav(float64x4_t x, float64x4_t y, float64x4_t z, double a, double b) {
//	return float64x4_t{a}*x*(y+float64x4_t{b}*z); 
       return a*x*(y+b*z);
}

float64x4_t dfmal(float64x4_t x, float64x4_t y, float64x4_t z, double a, double b) {
        float64x4_t r;
        for (int i=0; i!=4;++i)
          r[i] = a*x[i]*(y[i]+b*z[i]);
       return r;
}

float64x8_t dfma8v(float64x8_t x, float64x8_t y, float64x8_t z, double a, double b) {
        return a*x*(y+b*z);
}

float64x8_t dfma8l(float64x8_t x, float64x8_t y, float64x8_t z, double a, double b) {
        float64x8_t r;
        for (int i=0; i!=8;++i)
          r[i] = a*x[i]*(y[i]+b*z[i]);
       return r;
}

