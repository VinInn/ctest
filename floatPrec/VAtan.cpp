// #include "approx_exp.h"
// #include "approx_log.h"

#include<cmath>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<iostream>

#include <x86intrin.h>

typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;
typedef float __attribute__( ( vector_size( 32 ) ) ) float32x8_t;
typedef int __attribute__( ( vector_size( 32 ) ) ) int32x8_t;



template<typename Float>
inline
Float atan(Float t) {
  constexpr float PIO4F = 0.7853981633974483096f;
  constexpr Float zero = {0};
  Float z= (t > 0.4142135623730950f) ? (t-1.0f)/(t+1.0f) : t;
  Float ret = ( t > 0.4142135623730950f ) ? zero+PIO4F : zero;

  Float z2 = z * z;
  ret +=
    ((( 8.05374449538e-2f * z2
	- 1.38776856032E-1f) * z2
      + 1.99777106478E-1f) * z2
     - 3.33329491539E-1f) * z2 * z
    + z;

  return ret;
}


template<>
inline
float32x4_t atan<float32x4_t>(float32x4_t t) {
  constexpr float PIO4F = 0.7853981633974483096f;
  float32x4_t high = t > 0.4142135623730950f;
  auto z = t;
  float32x4_t ret={0.f,0.f,0.f,0.f};
    // if all low no need to blend
  if ( _mm_movemask_ps(high) != 0) {
    z   = ( t > 0.4142135623730950f ) ? (t-1.0f)/(t+1.0f) : t;
    ret = ( t > 0.4142135623730950f ) ? ret+PIO4F : ret;
  }
  // if( t > 0.4142135623730950f ) // * tan pi/8 
  
  auto z2 = z * z;
  ret +=
    ((( 8.05374449538e-2f * z2
	- 1.38776856032E-1f) * z2
      + 1.99777106478E-1f) * z2
     - 3.33329491539E-1f) * z2 * z
    + z;
  
  return  ret;
}



float32x4_t doAtan(float32x4_t z) { return atan(z);}


float32x4_t va[1024];
float32x4_t vb[1024];
float32x4_t vc[1024];

float a[4*1024];
float b[4*1024];


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


//inline
void computeV() {
  for (int i=0;i!=1024;++i)
    vb[i]=atan(va[i]);
}

//inline
void computeL() {
  for (int i=0;i!=4*1024;++i)
    b[i]=atan(a[i]);
}


// inline
void computeA() {
  for (int i=0;i!=1024;++i)
    vb[i]=apply(va[i],atan<float>);
}

// inline
void computeB() {
  for (int i=0;i!=1024;++i)
    for (int j=0;j!=4;++j)
    vb[i][j]=atan<float>(va[i][j]);
}



/*
void computeK() {
  for (int i=0;i!=1024;++i)
    vb[i]=apply(va[i],unsafe_logf<8>);
}
*/

#include<random>
#include<iostream>
std::mt19937 eng;
std::mt19937 eng2;
std::uniform_real_distribution<float> rgen(0.,1.);
 

void fillR() {
  for (int i=0;i!=1024;++i)
    va[i]=apply(va[i],[&](float){ return rgen(eng);});
}

void fillO() {
  float d=0;
  for (int i=0;i!=1024;++i)
    for (int j=0;j!=4;++j) {
      d+=0.8/(4.*1024.);
      va[i][j]=d;
    }
}

void fillW() {
  float d=0;
  float u=0.8;
  for (int i=0;i!=1024;++i)
    for (int j=0;j!=4;j+=2) {
      d+=0.8/(4.*1024.);
      u-=0.8/(4.*1024.);
      va[i][j]=d;
      va[i][j+1]=u;
    }
}


float sum() {
  float q=0;
  for (int i=0;i!=1024;++i) for (int j=0;j!=4;j+=2) { q+=vb[i][j];q-=vb[i][j+1];}
  return q;
}


unsigned int taux=0;
inline volatile unsigned long long rdtsc() {
 return __rdtscp(&taux);
}

int main(int argc, char**) {
  {
    long long t1=0;
    float s1=0;
    long long t2=0;
    float s2=0;
    long long t3=0;
    float s3=0;
    
    
    fillR();
    computeV();
    
    
    for (int i=0; i!=10000; ++i) {
      fillO();

      t1 -= rdtsc();
      computeV();
      t1 += rdtsc();
      s1+=sum();
      
      t2 -= rdtsc();
      computeA();
      t2 += rdtsc();
      s2+=sum();
      
      t3 -= rdtsc();
      computeB();
      t3 += rdtsc();
      s3+=sum();
      
    }
    std::cout << s1 << " " << double(t1)/10000 << std::endl;
    std::cout << s2 << " " << double(t2)/10000 << std::endl;
    std::cout << s3 << " " << double(t3)/10000 << std::endl << std::endl;
  }
  {
    long long t1=0;
    float s1=0;
    long long t2=0;
    float s2=0;
    long long t3=0;
    float s3=0;
    
    
    fillO();
    computeV();
    
    
    for (int i=0; i!=10000; ++i) {
      fillR();
      t1 -= rdtsc();
      computeV();
      t1 += rdtsc();
      s1+=sum();
      
      t2 -= rdtsc();
      computeA();
      t2 += rdtsc();
      s2+=sum();
      
      memcpy(a,va,1024*4);
      t3 -= rdtsc();
      computeL();
      t3 += rdtsc();
      memcpy(vb,b,1024*4);
      s3+=sum();
      
    }
    std::cout << s1 << " " << double(t1)/10000 << std::endl;
    std::cout << s2 << " " << double(t2)/10000 << std::endl;
    std::cout << s3 << " " << double(t3)/10000 << std::endl << std::endl;
  }
  {
    long long t1=0;
    float s1=0;
    long long t2=0;
    float s2=0;
    long long t3=0;
    float s3=0;
    
    
    fillO();
    computeV();
    
    
    for (int i=0; i!=10000; ++i) {
      fillW();
      t1 -= rdtsc();
      computeV();
      t1 += rdtsc();
      s1+=sum();
      
      t2 -= rdtsc();
      computeA();
      t2 += rdtsc();
      s2+=sum();
      
      t3 -= rdtsc();
      computeB();
      t3 += rdtsc();
      s3+=sum();
      
    }
    std::cout << s1 << " " << double(t1)/10000 << std::endl;
    std::cout << s2 << " " << double(t2)/10000 << std::endl;
    std::cout << s3 << " " << double(t3)/10000 << std::endl << std::endl;
  }
  return 0;
    
}
