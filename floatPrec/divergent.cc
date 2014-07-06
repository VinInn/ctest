#include<cmath>
float a[1024];
float b[1024];
float c[1024];

#define likely(x) (__builtin_expect(x, true))

void compute() {
  for (int i=0;i!=1024;++i) {
    if likely(a[i]<b[i])
	       c[i]=a[i]+b[i];
    else
      c[i]=std::sqrt(a[i]-b[i]);
  }
}

void computeB() {
  bool t[1024];
  for (int i=0;i!=1024;++i) {
    t[i]= a[i]<b[i];
    c[i]=a[i]+b[i];
  }
  for (int i=0;i!=1024;++i)
    if(t[i]) c[i]=std::sqrt(a[i]-b[i]);
}




#include <x86intrin.h>

typedef float __attribute__( ( vector_size( 32 ) ) ) float32x8_t;
typedef int __attribute__( ( vector_size( 32 ) ) ) int32x8_t;


float32x8_t va[1024];
float32x8_t vb[1024];
float32x8_t vc[1024];

void computeV() {
  for (int i=0;i!=1024;++i) {
    float32x8_t mask = va[i]<vb[i];
    if likely(_mm256_movemask_ps(mask) == 255) {
	vc[i]=va[i]+vb[i];
      } else {
      vc[i]= va[i]<vb[i] ? va[i]+vb[i] : va[i]-vb[i];
    }
  }
}
