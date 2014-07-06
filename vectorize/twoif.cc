#include<cmath>
float a[1024];
float b[1024];


void ok() {
#pragma omp simd
  for (int i=0;i<1024;++i) {
    a[i]=b[i];
  }
}

void bah() {
#pragma omp simd
  for (auto x : a) x+=2;

}

void bar(){
#pragma omp simd
  for (int i=0;i<1024;++i) {
    auto z = a[i];
    if (a[i] > 3.14f) z-=1.f;
    b[i] = 1.f/std::sqrt(z);
    if (a[i] > 3.14f) b[i]-=1.f;
  }
}

void foo(){
#pragma omp simd
  for (int i=0;i<1024;++i) {
    auto z = a[i];
    if (a[i] > 3.14f) z-=1.f;
    b[i] = 1.f/std::sqrt(z);
    if (a[i] > 1.f) b[i]-=1.f;
  }
}

