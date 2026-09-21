// nvcc -gencode arch=compute_75,code=sm_75 -O3 --expt-relaxed-constexpr -std=c++23 testClockSoA.cu -DNT=512 -DNB=4 -DMX=10000
#include "clockSoA.h"


template<typename T>
struct SoA {
   using Type = T;
   T * x;
   T * y;
   T * z;

};

template<typename T>
struct Q {
   constexpr void operator()(SoA<T>  y, SoA<T> const x, int j, int k) { 
     if constexpr (std::is_floating_point<T>::value)  {
       y.x[j] =  (y.x[j]*T(1.e-12))+(x.x[j]+x.y[j]*x.z[j]);
       y.y[j] =  (y.y[j]*T(1.e-12))+(x.y[j]+x.y[j]*x.x[j]); 
       y.z[j] =  (y.z[j]*T(1.e-12))+(x.z[j]+x.x[j]*x.y[j]);
     } else  {
       y.x[j] =  (y.x[j]>>15)+(x.x[j]+x.y[j]*x.z[j]);
       y.y[j] =  (y.y[j]>>15)+(x.y[j]+x.y[j]*x.x[j]);
       y.z[j] =  (y.z[j]>>15)+(x.z[j]+x.x[j]*x.y[j]);
     }
   }
};

template<typename T>
struct G {
  void operator()(SoA<T> & x, SoA<T> & y, int n) { 
    cudaMalloc(&a, 3*n * sizeof(T)); 
    cudaMalloc(&b, 3*n * sizeof(T));
    x.x = a; x.y = a+n; x.z = x.y + n;
    y.x = b; y.y = b+n; y.z = y.y + n;
  }
  ~G() {
    cudaFree(a);
    cudaFree(b);
  }
  T * a;
  T * b;
};




int main() {

   doClockSoA<G<double>,Q<double>,SoA<double>,SoA<double>>("",1000000);
   doClockSoA<G<float>,Q<float>,SoA<float>,SoA<float>>("",1000000);
   doClockSoA<G<int16_t>,Q<int16_t>,SoA<int16_t>,SoA<int16_t>>("",1000000);
   return 0;
}
