int whoamI;

#ifdef FAT
#pragma omp declare simd notinbranch
float __attribute__ ((__target__ ("default")))
fma(float x,float y, float z);
#pragma omp declare simd notinbranch
float __attribute__ ((__target__ ("arch=haswell")))
fma(float x,float y, float z);
#pragma omp declare simd notinbranch
float __attribute__ ((__target__ ("arch=bdver1")))
fma(float x,float y, float z);
#else
#pragma omp declare simd notinbranch
float
fma(float x,float y, float z);
#endif


float v1[1024],  v3[1024],  v2[1024],  v0[1024];

void foo() {
  #pragma omp simd safelen(8)
  for (int i=0; i<1024; ++i)
   v0[i] = fma(v1[i],v2[i],v3[i]);
}

#include<iostream>

int main(int argc, char ** argv) {
  whoamI=0;

  foo();
float s=0;
for (int i=0; i<1024; ++i) s+=v0[i];
  std::cout << s << std::endl;

  std::cout << whoamI << std::endl;

  return 0;
}
