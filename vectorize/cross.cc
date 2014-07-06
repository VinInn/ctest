#include<type_traits>

typedef float __attribute__( ( vector_size( 8 ) ) ) float32x2_t;
typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;
typedef double __attribute__ ((aligned(16)))  __attribute__( ( vector_size( 16 ) ) ) float64x2_t;
typedef double __attribute__ ((aligned(16)))  __attribute__( ( vector_size( 32 ) ) ) float64x4_t;
typedef double __attribute__( ( vector_size( 32 ) ) ) float32x8_t;
typedef double __attribute__( ( vector_size( 64 ) ) ) float64x8_t;
typedef long long __attribute__( ( vector_size( 32 ) ) ) int64x4_t;


// template<typename T, int N> using ExtVec =  T __attribute__( ( vector_size( N*sizeof(T) ) ) );

template<typename T, int N>
struct ExtVecTraits {
typedef T __attribute__ ((aligned(16))) __attribute__( ( vector_size( N*sizeof(T) ) ) ) type;
};

template<typename T, int N> using ExtVec =  typename ExtVecTraits<T,N>::type;


template<typename T> using Vec4 = ExtVec<T,4>;
template<typename T> using Vec2 = ExtVec<T,2>;

template<typename Vec> 
inline
Vec cross3(Vec x, Vec y) {
  //  typedef Vec4<T> Vec;
  // yz - zy, zx - xz, xy - yx, 0
  Vec x1200 = (Vec){ x[1], x[2], x[0], x[0] };
  Vec y2010 = (Vec){ y[2], y[0], y[1], y[0] };
  Vec x2010 = (Vec){ x[2], x[0], x[1], x[0] };
  Vec y1200 = (Vec){ y[1], y[2], y[0], y[0] };
  return x1200 * y2010 - x2010 * y1200;
}

template<typename V> 
inline
auto cross2(V x, V y) ->typename std::remove_reference<decltype(x[0])>::type {
  return x[0]*y[1]-x[1]*y[0];
}

template<typename V>
inline
auto dot(V  x, V y) ->typename  std::remove_reference<decltype(x[0])>::type {
  typedef typename std::remove_reference<decltype(x[0])>::type T;
  constexpr int N = sizeof(V)/sizeof(T);
  T ret=0;
  for (int i=0;i!=N;++i) ret+=x[i]*y[i];
  return ret;
}


typedef Vec2<float> Vec2F;
typedef Vec4<float> Vec4F;
typedef Vec4<float> Vec3F;
typedef Vec2<double> Vec2D;
typedef Vec4<double> Vec3D;
typedef Vec4<double> Vec4D;


//-----------------------

Vec4F csf(Vec4F x, Vec4F y) {
  return cross3(x,y);
}
Vec4D csf(Vec4D x, Vec4D y) {
  return cross3(x,y);
}

float dp(Vec4F x, Vec4F y) {
  return dot(x,y);
}
double dp(Vec4D x, Vec4D y) {
  return dot(x,y);
}



/*
template<typename Vec>
inline
Vec cross_product(Vec x, Vec y) {
  // yz - zy, zx - xz, xy - yx, 0
  Vec x1200 = (Vec){ x[1], x[2], x[0], x[0] };
  Vec y2010 = (Vec){ y[2], y[0], y[1], y[0] };
  Vec x2010 = (Vec){ x[2], x[0], x[1], x[0] };
  Vec y1200 = (Vec){ y[1], y[2], y[0], y[0] };
  return x1200 * y2010 - x2010 * y1200;
}

typedef Vec4<float> vf;
vf cross_productf(vf x, vf y) {
  return cross_product(x,y);
}

*/

float64x4_t cross_product(float64x4_t x, float64x4_t y) {
  // yz - zy, zx - xz, xy - yx, 0
  int64x4_t m1200{1,2,0,0};
  int64x4_t m2010{2,0,1,0};
  float64x4_t x1200 = __builtin_shuffle(x,m1200);
  float64x4_t y1200 = __builtin_shuffle(y,m1200);             
  float64x4_t x2010 = __builtin_shuffle(x,m2010);
  float64x4_t y2010 = __builtin_shuffle(y,m2010);             
  return x1200 * y2010 - x2010 * y1200;
}


/*
float dot_product(float32x4_t x, float32x4_t y) {
  float32x4_t res = x*y;
  float ret=0;
  for (int i=0;i!=4;++i) ret+=res[i];
  return ret;
}

float dot_product2(float32x4_t x, float32x4_t y) {
  float ret=0;
  for (int i=0;i!=4;++i) ret+=x[i]*y[i];
  return ret;
}


double dot_product(float64x4_t x, float64x4_t y) {
  float64x4_t res = x*y;
  double ret=0;
  for (int i=0;i!=4;++i) ret+=res[i];
  return ret;
}

double dot_product2(float64x4_t x, float64x4_t y) {
  double ret=0;
  for (int i=0;i!=4;++i) ret+=x[i]*y[i];
  return ret;
}
*/
