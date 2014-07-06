#include<cmath>

constexpr int N=4;
float  origin[N], boxsize[N];

void contains_v(float const * point, int *  isin, int np ) {
#pragma omp simd aligned( point, isin : 32)
  for(int k=0; k < np; ++k)    
    for (int dir=0; dir<N; ++dir)
      isin[k] += std::abs(point[N*k+dir]-origin[dir])<boxsize[dir];
}

void contains_b(float const *  point, bool *  isin, int np ) {
#pragma omp simd aligned( point, isin : 32)
     for(int k=0; k<  np; ++k)    for (int dir=0; dir<N; ++dir)
       isin[k] &= std::abs(point[N*k+dir]-origin[dir])<boxsize[dir];
}



typedef double __attribute__( ( vector_size( 32 ) ) ) float64x4_t;
typedef long long __attribute__( ( vector_size( 32 ) ) ) int64x4_t;
typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;
typedef int   __attribute__( ( vector_size( 16 ) ) ) int32x4_t;


float64x4_t ori, s;

inline
bool contains(float64x4_t point) {
  int64x4_t res =  ( point> ori-s & point< ori+s );
  return res[0]+res[1]+res[2]==3;
}

float64x4_t pointsD[1024];
bool ins[1024];

void containsD() {
   for(int k=0; k != 1024; ++k)
    ins[k] = contains(pointsD[k]);
}


float32x4_t orif, sf;

bool contains(float32x4_t point) {
  int32x4_t res =  ( point> orif-sf & point< orif+sf );
  return res[0]+res[1]+res[2]==3;
}


float32x4_t pointsF[1024];

void containsF() {
   for(int k=0; k != 1024; ++k) 
    ins[k] = contains(pointsF[k]);
}
