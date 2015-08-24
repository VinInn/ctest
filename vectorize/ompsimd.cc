#pragma omp declare simd notinbranch
template<int N>
float foo(float);


#pragma omp declare simd inbranch
template<int N>
float foo(float,float);


float v0[1024];
float v1[1024];
float v2[1024];
float v3[1024];

void go() {
 #pragma omp simd
 for(int i=0; i<1024; ++i) {
   v0[i] = foo<2>(v2[i]);
 }

}


void gone() {
 #pragma omp simd
 for(int i=0; i<1024; ++i) {
   auto f = foo<2>(v2[i]);
   if(0.f>v0[i]) 
    v0[i] = f;
//   else v1[i] = foo<2>(v2[i]);
 }

}


