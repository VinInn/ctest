// nvcc -gencode arch=compute_75,code=sm_75 -O3 --expt-relaxed-constexpr -std=c++23 testClockSoA.cu -DNT=512 -DNB=4 -DMX=10000
// ./a.out | grep gtime | cut -d' ' -f6 | tr '\n' ','
#include "clockSoA.h"
#include <cassert>

template<typename T>
struct SoA {
   using Type = T;
   T * x;
   T * y;
   T * z;

};

// 1D 
template<typename T>
struct Q {
   constexpr void operator()(SoA<T>  y, SoA<T> const x, int const * ind, int j, int k, int n) { 
     if constexpr (std::is_floating_point<T>::value)  {
      auto v = (y.x[j]*T(1.e-12))+x.x[j];
      auto w = (y.y[j]*T(1.e-12))+x.y[j];
      auto u = (y.z[j]*T(1.e-12))+x.z[j];
       y.x[j] =  v+w*u;
       y.y[j] =  w+v*u; 
       y.z[j] =  u+v*w;
     } else  {
       auto v = float((y.x[j]>>15)+x.x[j]);
       auto w = float((y.y[j]>>15)+x.y[j]);
       auto u = float((y.z[j]>>15)+x.z[j]);
       y.x[j] =  v+w*u;
       y.y[j] =  w+v*u;
       y.z[j] =  u+v*w;
     }
   }
};



template<typename T>
struct R {
   constexpr void operator()(SoA<T>  y, SoA<T> const x, int const * ind, int i, int k, int n) {
     int j = ind[i];
     // assert(j>=0); assert(j<n);
     if constexpr (std::is_floating_point<T>::value)  {
      auto v = (y.x[i]*T(1.e-12))+x.x[j];
      auto w = (y.y[i]*T(1.e-12))+x.y[j];
      auto u = (y.z[i]*T(1.e-12))+x.z[j];
       y.x[i] =  v+w*u;
       y.y[i] =  w+v*u;
       y.z[i] =  u+v*w;
     } else  {
       auto v = float((y.x[i]>>15)+x.x[j]);
       auto w = float((y.y[i]>>15)+x.y[j]);
       auto u = float((y.z[i]>>15)+x.z[j]);
       y.x[i] =  v+w*u;
       y.y[i] =  w+v*u;
       y.z[i] =  u+v*w;
     }
   }
};


// 1D combi
template<typename T, bool L>
struct W {
   constexpr void operator()(SoA<T>  y, SoA<T> const x, int const * ind, int j, int k, int n) {
     int a; 
     int b;
     if constexpr (L) {
       a = max(j-16,0);
       b = min(j+16,n);
     } else {
       int s = n/2;
       a = (j<s) ? max(s+j-16,0) : max(j-s-16,0);
       b = (j<s) ? min(s+j+16,n) : min(j-s+16,n);
     }
     for (int  i =a; i<b; ++i) {
     if constexpr (std::is_floating_point<T>::value)  {
       y.x[j] =  (y.x[j]*T(1.e-12))+(x.x[j]+x.y[i]*x.z[j]);
       y.y[j] =  (y.y[j]*T(1.e-12))+(x.y[j]+x.y[j]*x.x[i]);
       y.z[j] =  (y.z[j]*T(1.e-12))+(x.z[i]+x.x[j]*x.y[j]);
     } else  {
       y.x[j] =  (y.x[j]>>15)+int(float(x.x[j])+float(x.y[i])*float(x.z[j]));
       y.y[j] =  (y.y[j]>>15)+int(float(x.y[j])+float(x.y[j])*float(x.x[i]));
       y.z[j] =  (y.z[j]>>15)+int(float(x.z[i])+float(x.x[j])*float(x.y[j]));
     }
     }
   }
};


#include<random>
template<typename T>
struct G {
  int * operator()(SoA<T> & x, SoA<T> & y, int n) { 
    cudaMalloc(&a, 3*n * sizeof(T)); 
    cudaMalloc(&b, 3*n * sizeof(T));
    cudaMalloc(&ind, n * sizeof(int));
    x.x = a; x.y = a+n; x.z = x.y + n;
    y.x = b; y.y = b+n; y.z = y.y + n;

    int l[n];
    std::random_device rd;  // a seed source for the random number engine
    std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> rint(0,n);
    for (int i=0; i<n;++i) l[i]=rint(gen);
    cudaMemcpy(ind, l, sizeof(x),cudaMemcpyHostToDevice);
    return ind;
  }
  ~G() {
    cudaFree(a);
    cudaFree(b);
    cudaFree(ind);
  }
  T * a;
  T * b;
  int * ind;
};




int main() {

   int n = 256*1024;
   doClockSoA<G<double>,Q<double>,SoA<double>,SoA<double>>("d l",n);
   doClockSoA<G<float>,Q<float>,SoA<float>,SoA<float>>("f l",n);
   doClockSoA<G<int16_t>,Q<int16_t>,SoA<int16_t>,SoA<int16_t>>("i l",n);
   doClockSoA<G<double>,Q<double>,SoA<double>,SoA<double>,2>("dl ",n);
   doClockSoA<G<float>,Q<float>,SoA<float>,SoA<float>,2>("f l",n);
   doClockSoA<G<int16_t>,Q<int16_t>,SoA<int16_t>,SoA<int16_t>,2>("i l",n);
   doClockSoA<G<double>,Q<double>,SoA<double>,SoA<double>,4>("d l",n);
   doClockSoA<G<float>,Q<float>,SoA<float>,SoA<float>,4>("f l",n);
   doClockSoA<G<int16_t>,Q<int16_t>,SoA<int16_t>,SoA<int16_t>,4>("i l",n);


   doClockSoA<G<double>,R<double>,SoA<double>,SoA<double>>("d r",n);
   doClockSoA<G<float>,R<float>,SoA<float>,SoA<float>>("f r",n);
   doClockSoA<G<int16_t>,R<int16_t>,SoA<int16_t>,SoA<int16_t>>("i r",n);
   doClockSoA<G<double>,R<double>,SoA<double>,SoA<double>,2>("d r",n);
   doClockSoA<G<float>,R<float>,SoA<float>,SoA<float>,2>("f t",n);
   doClockSoA<G<int16_t>,R<int16_t>,SoA<int16_t>,SoA<int16_t>,2>("i r",n);



   doClockSoA<G<double>,W<double,true>,SoA<double>,SoA<double>>("",n);
   doClockSoA<G<float>,W<float,true>,SoA<float>,SoA<float>>("",n);
   doClockSoA<G<int16_t>,W<int16_t,true>,SoA<int16_t>,SoA<int16_t>>("",n);
   doClockSoA<G<double>,W<double,false>,SoA<double>,SoA<double>>("",n);
   doClockSoA<G<float>,W<float,false>,SoA<float>,SoA<float>>("",n);
   doClockSoA<G<int16_t>,W<int16_t,false>,SoA<int16_t>,SoA<int16_t>>("",n);

   return 0;
}
