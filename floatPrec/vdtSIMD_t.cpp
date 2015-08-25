#include "vdtSIMD.h"


float v0[1024];
float v1[1024];
float v2[1024];
float v3[1024];
float v4[1024];



void go() {
#pragma omp simd
 for (int i=0; i<1024; ++i) { 
     v0[i] = vdt::simd_atan2f(v1[i],v2[i]);
     v4[i] = vdt::simd_logf(v3[i]);
 }
}


#include<iostream>

int main(int argc, char** ) {

  v1[0] = v2[0] = v3[0] = 0.5;

  go();

  std::cout << v0[0] << ' ' << v4[0] << std::endl;
}
