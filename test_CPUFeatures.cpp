// #pragma GCC target ("sse2")


#include <cpuid.h>
#include <x86intrin.h>
#include<algorithm>
#include<iostream>


float  __attribute__ ((__target__ ("sse2"))) sum0(float const * __restrict__ x, float const * __restrict__ y, float const * __restrict__ z) {
  float sum=0;
  for (int i=0; i!=1024; ++i)
    sum += z[i]+x[i]*y[i];
  return sum;
}

float  __attribute__ ((__target__ ("sse3"))) sum(float const * __restrict__ x, float const * __restrict__ y, float const * __restrict__ z) {
  float sum=0;
  for (int i=0; i!=1024; ++i)
    sum += z[i]+x[i]*y[i];
  return sum;
}

float  __attribute__ ((__target__ ("fma4"))) sumf(float const * __restrict__ x, float const * __restrict__ y, float const * __restrict__ z) {
  float sum=0;
  for (int i=0; i!=1024; ++i)
    sum += z[i]+x[i]*y[i];
  return sum;
}


class CPUFeatures {

public:


  enum feature {x86, sse, sse2, sse3, ssse3, sse41, sse42, avx, fma4 };
  

  static feature get() {
      unsigned int eax, ebx, ecx, edx;

    __cpuid (1, eax, ebx, ecx, edx);
    
    if (ecx & bit_FMA4)
      return fma4;
    if (ecx & bit_AVX)
      return avx;
    if (ecx & bit_SSE4_2)
      return sse42;
    if (ecx & bit_SSE4_1)
      return sse41;
    if (ecx & bit_SSSE3)
      return ssse3;
    if (ecx & bit_SSE3)
      return sse3;
    if (edx & bit_SSE2)
      return sse2;
    if (edx & bit_SSE)
      return sse;
    return x86;
  }
  
};


template<int F>
struct CPU{
  enum {value=F};
};

CPUFeatures::feature cpuFeature() {
  static CPUFeatures::feature f= CPUFeatures::get();
  return f;
}

template<typename F, typename... Args>
struct IFunc {
  F f;
  // typedef typename std::result_of<F(Args...)>::type return_type;
  typedef typename std::result_of<F(float)>::type return_type;
  return_type operator()(Args... args) const {
    CPUFeatures::feature ff = cpuFeature();
    switch (ff) {
    case CPUFeatures::avx :
      return  f_avx(std::forward<Args>(args)...);
    case CPUFeatures::sse42 :
      return  f_sse42(std::forward<Args>(args)...);
    case CPUFeatures::sse41 :
      return  f_sse41(std::forward<Args>(args)...);
    case CPUFeatures::sse3 :
      return  f_sse3(std::forward<Args>(args)...);
    default:
      return f_x86(std::forward<Args>(args)...);
    }
    return f_x86(std::forward<Args>(args)...);
  }

  return_type __attribute__ ((__target__ ("avx"))) f_avx(Args... args) const {
    std::cout << "avx " << cpuFeature() << std::endl;
    return f.call(CPU<CPUFeatures::avx>(),std::forward<Args>(args)...);
  }

  return_type __attribute__ ((__target__ ("sse4.2"))) f_sse42(Args... args) const {
    std::cout << "sse4.2 " << cpuFeature() << std::endl;
    return f.call(CPU<CPUFeatures::sse42>(),std::forward<Args>(args)...);
  }

  return_type __attribute__ ((__target__ ("sse4.1"))) f_sse41(Args... args) const {
    std::cout << "sse4.1 " << cpuFeature() << std::endl;
    return f.call(CPU<CPUFeatures::sse41>(),std::forward<Args>(args)...);    
  }
  return_type __attribute__ ((__target__ ("sse3"))) f_sse3(Args... args) const {
    std::cout << "sse3 " << cpuFeature() << std::endl;
    return f.call(CPU<CPUFeatures::sse3>(),std::forward<Args>(args)...);
  }
  return_type f_x86(Args... args) const {
    std::cout << "x86 " << cpuFeature() << std::endl;
    return f.call(CPU<CPUFeatures::x86>(),std::forward<Args>(args)...);
  }
};

#ifdef __SSE4_2__
#define  IFUNC_SSE4_2
#elif  defined(__SSE4_1__)
#define  IFUNC_SSE4_1
#elif defined(__SSE3__)
#define  IFUNC_SSE3
#endif

struct F__hi {
  inline float __attribute__ ((visibility ("internal"))) operator()(float f) const;
  template<typename F>
  inline float __attribute__ ((visibility ("internal"))) call(F, float f) const {
#ifdef IFUNC_SSE4_2
    std::cout << "sse4.2 " << cpuFeature() << std::endl;
#elif defined(IFUNC_SSE4_1)
    std::cout << "sse4.1 " << cpuFeature() << std::endl;
#elif defined(IFUNC_SSE3)
    std::cout << "sse3 " << cpuFeature() << std::endl;
#else
    std::cout << "sse2 " << cpuFeature() << std::endl;
#endif
    return f;
  }
};


inline float hi(float v) {
  IFunc<F__hi,float> f;
  return f(v);
}


int main() {
  CPU<CPUFeatures::x86> c;
  std::cout <<  CPU<CPUFeatures::x86>::value << std::endl;
  F__hi f;
  f.call(CPU<CPUFeatures::x86>(),5.f);
  std::cout << "hi" << std::endl;

  hi(3);

}
