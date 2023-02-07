#include <cstdio>

#include <xmmintrin.h>
#include<pmmintrin.h>

int main() {

typedef union { unsigned int n; float x; } union_t;

//  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
//  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);

//  #pragma omp parallel for schedule(dynamic,32)
  for (int i=0; i<10; ++i) {
    union_t u;
    u.n = i;
    float x = u.x;
    printf("%d %a\n",i,x);
  }

  return 0;
}

