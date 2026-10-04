// nvcc -gencode arch=compute_75,code=sm_75 -O3 --expt-relaxed-constexpr -std=c++23 testClockSoA.cu -DNT=512 -DNB=4 -DMX=10000
// ./a.out | grep gtime | cut -d' ' -f6 | tr '\n' ','; echo ']\n,['
#include "clockSoA.h"
#include <cassert>
#include <cmath>
#include <cuda_fp16.h>
#include <bit>

#ifndef HOST_DEVICE_CONSTANT
#ifdef __CUDA_ARCH__
#define HOST_DEVICE_CONSTANT __device__ constexpr
#else
#define HOST_DEVICE_CONSTANT constexpr
#endif
#endif

#ifdef __NVCC__
#define HD_INLINE __device__ __host__ inline
#else
#define HD_INLINE inline
#endif



template<typename T>
struct SoA {
   using Type = T;
   T * __restrict__ x;
   T * __restrict__ y;
   T * __restrict__ z;

#ifdef CONSTM
  __device__ __forceinline__
   T xc(int i) const { return __ldg(x+i);}
  __device__ __forceinline__
   T yc(int i) const { return __ldg(y+i);}
   __device__ __forceinline__
   T zc(int i) const { return __ldg(z+i);}
#else
  __device__ __forceinline__
   T xc(int i) const { return *(x+i);}
  __device__ __forceinline__
   T yc(int i) const { return *(y+i);}
   __device__ __forceinline__
   T zc(int i) const { return *(z+i);}
#endif

};


template<typename T>
struct poly {
   using Float=T;
   HD_INLINE T operator()(T x) {
     x = std::abs(x);
     return
     T(0.999999986449) + x*(T(2.06961523705e-6) + x*( T(-0.500073072319) + x*( T(0.00106750423055) + x*( T(0.200707822535) + x*( T(0.0295750777896) + x*( T(-0.150307883454) + x*( T(0.0821370891714) + x*T(-0.0150543655971)
      )))))));
   }
};

template<typename T>
struct U {
   using Float=T;
   HD_INLINE T operator()(T x) { return x;}
};


constexpr uint32_t shift15(uint32_t x) {
    return (x>>15)&(0x00010001);
}
constexpr std::tuple<float,float>  unpack(uint32_t x) {
    uint16_t uj = x&0xFFFF;
    uint16_t ui = (x&0xFFFF0000)>>16;
    return {std::bit_cast<int16_t>(ui),std::bit_cast<int16_t>(uj)};
}
constexpr uint32_t pack(float a1, float a2) {
  int16_t i = a1; int16_t j = a2;  // should be round
  uint32_t ui = std::bit_cast<uint16_t>(i);
  uint32_t uj = std::bit_cast<uint16_t>(j);
  return (ui<<16) & uj;
}


// 1D 
template<typename T, typename F>
struct Q {
   constexpr void operator()(SoA<T>  y, SoA<T> const x, int const * ind, int j, int k, int n) { 
     F f;   
     if constexpr (std::is_floating_point<T>::value 
                   || std::is_same<half, typename std::remove_cv<T>::type>::value
                   || std::is_same<half2, typename std::remove_cv<T>::type>::value)  {
       if constexpr (std::is_same<half2, typename std::remove_cv<T>::type>::value) {
        auto v = __hfma2(y.x[j],T(1.e-12,1.e-12),x.xc(j));
        auto w = __hfma2(y.y[j],T(1.e-12,1.e-12),x.yc(j));
        auto u = __hfma2(y.z[j],T(1.e-12,1.e-12),x.zc(j));
        y.x[j] =  f(__hfma2(w,u,v));
        y.y[j] =  f(__hfma2(v,u,w));
        y.z[j] =  f(__hfma2(v,w,u));
       }else {
        auto v = (y.x[j]*T(1.e-12))+x.xc(j);
        auto w = (y.y[j]*T(1.e-12))+x.yc(j);
        auto u = (y.z[j]*T(1.e-12))+x.zc(j);
        y.x[j] =  f(v+w*u);
        y.y[j] =  f(w+v*u);
        y.z[j] =  f(u+v*w);
       }
     } else if constexpr (std::is_same<uint32_t, typename std::remove_cv<T>::type>::value) {
       auto iv = y.x[j]; iv = __vadd2(shift15(iv),x.xc(j));
       auto iw = y.y[j]; iw = __vadd2(shift15(iw),x.yc(j));
       auto iu = y.z[j]; iu = __vadd2(shift15(iu),x.zc(j));
       auto [v1,v2] = unpack(iv);
       auto [w1,w2] = unpack(iw);
       auto [u1,u2] = unpack(iu);
       iv = pack(f(v1+w1*u1),f(v2+w2*u2));
       iw = pack(f(w1+v1*u1),f(w2+v2*u2));
       iu = pack(f(u1+v1*w1),f(u2+v2*w2));
       y.x[j] =  iv;
       y.y[j] =  iw;
       y.z[j] =  iu;
     } else {  // int16
       auto v = float((y.x[j]>>15)+x.xc(j));
       auto w = float((y.y[j]>>15)+x.yc(j));
       auto u = float((y.z[j]>>15)+x.zc(j));
       y.x[j] =  f(v+w*u);
       y.y[j] =  f(w+v*u);
       y.z[j] =  f(u+v*w);

     }
   }
};





template<typename T, typename F>
struct R {
   constexpr void operator()(SoA<T>  y, SoA<T> const x, int const * ind, int i, int k, int n) {
     F f;
     int j = ind[i];
     // assert(j>=0); assert(j<n);
    if constexpr (std::is_floating_point<T>::value
                   || std::is_same<half, typename std::remove_cv<T>::type>::value
                   || std::is_same<half2, typename std::remove_cv<T>::type>::value)  {
      auto v = (y.x[i]*T(1.e-12))+x.xc(j);
      auto w = (y.y[i]*T(1.e-12))+x.yc(j);
      auto u = (y.z[i]*T(1.e-12))+x.zc(j);
       y.x[i] =  f(v+w*u);
       y.y[i] =  f(w+v*u);
       y.z[i] =  f(u+v*w);
     } else  {
       auto v = float((y.x[i]>>15)+x.xc(j));
       auto w = float((y.y[i]>>15)+x.yc(j));
       auto u = float((y.z[i]>>15)+x.zc(j));
       y.x[i] =  f(v+w*u);
       y.y[i] =  f(w+v*u);
       y.z[i] =  f(u+v*w);
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
       y.x[j] =  (y.x[j]*T(1.e-12))+(x.xc(j)+x.yc(i)*x.zc(j));
       y.y[j] =  (y.y[j]*T(1.e-12))+(x.yc(j)+x.yc(j)*x.xc(i));
       y.z[j] =  (y.z[j]*T(1.e-12))+(x.zc(i)+x.xc(j)*x.yc(j));
     } else  {
       y.x[j] =  (y.x[j]>>15)+int(float(x.xc(j))+float(x.yc(i))*float(x.zc(j)));
       y.y[j] =  (y.y[j]>>15)+int(float(x.yc(j))+float(x.yc(j))*float(x.xc(i)));
       y.z[j] =  (y.z[j]>>15)+int(float(x.zc(i))+float(x.xc(j))*float(x.yc(j)));
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
    T q[3*n];
    std::random_device rd;  // a seed source for the random number engine
    std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> rint(0,n);
    std::uniform_real_distribution<float> rf(-1.0,1.0);
    std::uniform_int_distribution<> rint16(-32768,-32768);
#ifdef NRAND
    for (int i=0; i<n;++i) l[i]=i;
#else
    for (int i=0; i<n;++i) l[i]=rint(gen);
#endif
    for (int i=0; i<3*n;++i) {
      if constexpr (std::is_floating_point<T>::value || std::is_same<half, typename std::remove_cv<T>::type>::value)
        q[i] = rf(gen);
      else if constexpr(std::is_same<half2, typename std::remove_cv<T>::type>::value)
         q[i] = half2{rf(gen), rf(gen)};
      else
        q[i] = rint16(gen);
    }
    cudaMemcpy(ind, l, n*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(a, q, 3*n*sizeof(T),cudaMemcpyHostToDevice);
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
   doClockSoA<G<double>,Q<double,U<double>>,SoA<double>,SoA<double>>("d l",n);
   doClockSoA<G<float>,Q<float,U<float>>,SoA<float>,SoA<float>>("f l",n);
   doClockSoA<G<int16_t>,Q<int16_t,U<float>>,SoA<int16_t>,SoA<int16_t>>("i l",n);
   doClockSoA<G<double>,Q<double,U<double>>,SoA<double>,SoA<double>,2>("d l ",n);
   doClockSoA<G<float>,Q<float,U<float>>,SoA<float>,SoA<float>,2>("f l",n);
   doClockSoA<G<int16_t>,Q<int16_t,U<float>>,SoA<int16_t>,SoA<int16_t>,2>("i l",n);

   doClockSoA<G<half>,Q<half,U<half>>,SoA<half>,SoA<half>,1>("h l",n);
   doClockSoA<G<half>,Q<half,U<half>>,SoA<half>,SoA<half>,2>("h l",n);
   doClockSoA<G<half2>,Q<half2,U<half2>>,SoA<half2>,SoA<half2>,1>("h2 l",n/2);
   doClockSoA<G<uint32_t>,Q<uint32_t,U<float>>,SoA<uint32_t>,SoA<uint32_t>,1>("i2 l",n/2);


   doClockSoA<G<double>,R<double,U<double>>,SoA<double>,SoA<double>>("d r",n);
   doClockSoA<G<float>,R<float,U<float>>,SoA<float>,SoA<float>>("f r",n);
   doClockSoA<G<int16_t>,R<int16_t,U<float>>,SoA<int16_t>,SoA<int16_t>>("i r",n);
   doClockSoA<G<double>,R<double,U<double>>,SoA<double>,SoA<double>,2>("d r",n);
   doClockSoA<G<float>,R<float,U<float>>,SoA<float>,SoA<float>,2>("f r",n);
   doClockSoA<G<int16_t>,R<int16_t,U<float>>,SoA<int16_t>,SoA<int16_t>,2>("i r",n);


   doClockSoA<G<double>,Q<double,poly<double>>,SoA<double>,SoA<double>>("d pl",n);
   doClockSoA<G<float>,Q<float,poly<float>>,SoA<float>,SoA<float>>("f pl",n);
   doClockSoA<G<int16_t>,Q<int16_t,poly<float>>,SoA<int16_t>,SoA<int16_t>>("i pl",n);
   doClockSoA<G<double>,Q<double,poly<double>>,SoA<double>,SoA<double>,2>("d pl ",n);
   doClockSoA<G<float>,Q<float,poly<float>>,SoA<float>,SoA<float>,2>("f pl",n);
   doClockSoA<G<int16_t>,Q<int16_t,poly<float>>,SoA<int16_t>,SoA<int16_t>,2>("i pl",n);


   doClockSoA<G<double>,R<double,poly<double>>,SoA<double>,SoA<double>>("d pr",n);
   doClockSoA<G<float>,R<float,poly<float>>,SoA<float>,SoA<float>>("f pr",n);
   doClockSoA<G<int16_t>,R<int16_t,poly<float>>,SoA<int16_t>,SoA<int16_t>>("i pr",n);
   doClockSoA<G<double>,R<double,poly<double>>,SoA<double>,SoA<double>,2>("d pr",n);
   doClockSoA<G<float>,R<float,poly<float>>,SoA<float>,SoA<float>,2>("f t",n);
   doClockSoA<G<int16_t>,R<int16_t,poly<float>>,SoA<int16_t>,SoA<int16_t>,2>("i pr",n);


   doClockSoA<G<double>,W<double,true>,SoA<double>,SoA<double>>("",n);
   doClockSoA<G<float>,W<float,true>,SoA<float>,SoA<float>>("",n);
   doClockSoA<G<int16_t>,W<int16_t,true>,SoA<int16_t>,SoA<int16_t>>("",n);
   doClockSoA<G<double>,W<double,false>,SoA<double>,SoA<double>>("",n);
   doClockSoA<G<float>,W<float,false>,SoA<float>,SoA<float>>("",n);
   doClockSoA<G<int16_t>,W<int16_t,false>,SoA<int16_t>,SoA<int16_t>>("",n);

   return 0;
}
