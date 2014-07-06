#include<cmath>
float __attribute__ ((aligned(16))) a[1024];
float __attribute__ ((aligned(16))) b[1024];

void  fV() {
  for (int i=0; i!=1024; ++i)
    b[i] =  std::floor(a[i]);
}
