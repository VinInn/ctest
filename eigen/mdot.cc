template<int N>
float dot(float const * __restrict__ a, float const * __restrict__ b) {
  return a[0]*b[0]+a[N]*b[N]+a[2*N]*b[2*N]+a[3*N]*b[3*N]+a[4*N]*b[4*N];
}


void vdot(float const * __restrict__ a, float const * __restrict__ b, float * __restrict__ r) {
   #pragma omp simd 
   for (int i=0; i<256; ++i) {
     r[i] = dot<1024>(a+i,b+i);
   }
}

void mdot(float const * __restrict__ a, float const * __restrict__ b, float * __restrict__ r) {
   constexpr int os = 5*1024;
//   #pragma omp simd
   for (int i=0; i<256; ++i) {
     r[i] = dot<1024>(a+i,b+i);
     r[i+1024] = dot<1024>(a+i+os,b+i);
     r[i+2*1024] = dot<1024>(a+i+2*os,b+i);
     r[i+3*1024] = dot<1024>(a+i+3*os,b+i);
     r[i+4*1024] = dot<1024>(a+i+4*os,b+i);
   }
}
