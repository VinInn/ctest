#include<iostream>
#include<string>

typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;
typedef float __attribute__( ( vector_size( 32 ) ) ) float32x8_t;
typedef float __attribute__( ( vector_size( 64 ) ) ) float32x16_t;

static std::string fathi;

#define FATHALLO(...) char const * __attribute__ ((__target__ (__VA_ARGS__))) \
  fathelloCPP() { fathi = std::string("targer is ")+__VA_ARGS__; return fathi.c_str();}


FATHALLO("default")
FATHALLO("sse3")
FATHALLO("arch=corei7")
FATHALLO("arch=bdver1")
FATHALLO("arch=core-avx2")
// FATHALLO("avx2","fma")
FATHALLO("arch=corei7-avx")
// FATHALLO("avx512f")
// FATHALLO()

extern "C" {
  char const * fathello() { return fathelloCPP();}
}


#define FATLIB(RET,FUN) RET __attribute__ ((__target__ ("default"))) FUN \
RET __attribute__ ((__target__ ("sse3"))) FUN \
RET __attribute__ ((__target__ ("arch=corei7"))) FUN \
RET __attribute__ ((__target__ ("arch=bdver1"))) FUN \
RET __attribute__ ((__target__ ("avx2","fma"))) FUN \
RET __attribute__ ((__target__ ("arch=corei7-avx"))) FUN \
RET __attribute__ ((__target__ ("avx512f"))) FUN


inline
float theFMA (float x, float y, float z) { return x+y*z;}

#define FATFMA myfmaCPP(float x, float y, float z) { return theFMA(x,y,z);} 
#define FATFMARETURN float

FATLIB(FATFMARETURN,FATFMA)

extern "C" {
  float myfma(float x, float y, float z) { return myfmaCPP(x,y,z);} 
}


float __attribute__ ((__target__ ("default")))
mySum(float vx, float vy) {
  return vx+vy;
}


float __attribute__ ((__target__ ("sse3")))
mySum(float vx, float vy) {
  return vx+vy;
}


float __attribute__ ((__target__ ("arch=nehalem")))
mySum(float vx, float vy) {
  return vx+vy;
}

float __attribute__ ((__target__ ("avx2","fma")))
mySum(float vx, float vy) {
  return vx+vy;
}


float __attribute__ ((__target__ ("avx512f")))
mySum(float vx, float vy) {
  return vx+vy;
}


float32x4_t __attribute__ ((__target__ ("default")))
mySum(float32x4_t vx, float32x4_t vy) {
  return vx+vy;
}

float32x4_t __attribute__ ((__target__ ("sse3")))
mySum(float32x4_t vx, float32x4_t vy) {
  return vx+vy;
}

float32x4_t __attribute__ ((__target__ ("arch=nehalem")))
mySum(float32x4_t vx, float32x4_t vy) {
  return vx+vy;
}

float32x4_t __attribute__ ((__target__ ("arch=haswell")))
mySum(float32x4_t vx, float32x4_t vy) {
  return vx+vy;
}

float32x4_t __attribute__ ((__target__ ("arch=bdver1")))
mySum(float32x4_t vx, float32x4_t vy) {
  return vx+vy;
}

/*
float32x4_t __attribute__ ((__target__ ("avx512f")))
mySum(float32x4_t vx, float32x4_t vy) {
  return vx+vy;
}
*/

/*

float32x8_t __attribute__ ((__target__ ("arch=haswell")))  
mySum(float32x8_t vx, float32x8_t vy) {
  return vx+vy;
}

float32x8_t  __attribute__ ((__target__ ("avx512f"))) 
mySum(float32x8_t vx, float32x8_t vy) {
  return vx+vy;
}

*/

/*
float32x16_t  __attribute__ ((__target__ ("sse3"))) 
mySum(float32x16_t vx, float32x16_t vy) {
  return vx+vy;
}
*/

/*
float32x16_t  __attribute__ ((__target__ ("avx512f"))) 
mySum(float32x16_t vx, float32x16_t vy) {
  return vx+vy;
}
*/


#ifdef MAIN
int main() {
  std::cout << fathello() << std::endl;
  
  float a=1, b=-1, c=4.5;
  float e= mySum(a,b);
  float d=myfma(a,b,c);

  float32x4_t x{0,1,2,3};
  float32x4_t y = x+1;

  float32x4_t z = mySum(x,y);

  return d*z[0]>e;
}
#endif
