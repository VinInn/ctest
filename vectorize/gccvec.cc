#include<cmath>

float v0[1024];
float v1[1024];
float v2[1024];
float v3[1024];


// #pragma omp declare simd notinbranch
// extern "C" float expf(float);

#pragma omp declare simd notinbranch
extern "C" void sincosf(float,float * s,float* c);


void cexp() {
  #pragma omp simd
  for(int i=0; i<1024; ++i) {
    v0[i] = expf(v2[i]);
  }
  
}

void vpow() {
  #pragma omp simd
  for(int i=0; i<1024; ++i) {
    v0[i] = powf(v2[i],v1[i]);
  }

}

void rot() {
  #pragma omp simd
  for(int i=0; i<1024; ++i) {
//    v1[i] = 0.5f*sinf(v2[i]) -0.5f*cosf(v2[i]);
    float c,s; sincosf(v2[i],&s,&c);
    v1[i] = 0.5f*s -0.5f*c;

  }

}

#ifndef NOMAIN

#include<iostream>

int main() {

 for(int i=0; i<1024; ++i) v2[i]=0.01*i;
 cexp();
 float s=0;
 for(int i=0; i<1024; ++i) s+=v0[i];
 std::cout << std::hexfloat << s << std::endl;

 rot();
 s=0;
 for(int i=0; i<1024; ++i) s+=v0[i];
 std::cout << std::hexfloat << s << std::endl;


 return 0;
}

#endif
