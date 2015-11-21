#ifndef NATIVE_VECTOR_H
#define NATIVE_VECTOR_H

#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>
#include <type_traits>
#include <cstring>
#include <x86intrin.h>

// only for float, easy to extend to double...
namespace nativeVector {

  typedef float __attribute__( ( vector_size(  4 ) ) ) float32x1_t;
  typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;
  typedef float __attribute__( ( vector_size( 32 ) ) ) float32x8_t;
  typedef float __attribute__( ( vector_size( 64 ) ) ) float32x16_t;

  typedef int __attribute__( ( vector_size(  4 ) ) ) int32x1_t;
  typedef int __attribute__( ( vector_size( 16 ) ) ) int32x4_t;
  typedef int __attribute__( ( vector_size( 32 ) ) ) int32x8_t;
  typedef int __attribute__( ( vector_size( 64 ) ) ) int32x16_t;


  template<typename T>
  struct IntType { using type = T; };
  template<>
  struct IntType<float> { using type = int; };
  template<>
  struct IntType<double> {  using type = long long;};
  template<typename T>
  struct UIntType { using type = T; };
  template<>
  struct UIntType<float> { using type = unsigned int; };
  template<>
  struct UIntType<double> {  using type = unsigned long long;};

  template<typename V>
  struct VType {
    static constexpr auto elem(V x) -> typename std::remove_reference<decltype(x[0])>::type { return x[0];}	      
  };
  template<>
  struct VType<float> {
    static constexpr auto elem(float x) -> float { return x;}	      
  };
  template<>
  struct VType<double> {
    static constexpr auto elem(double x) -> double { return x;}	      
  };

  
  template<typename VF>
  struct ConvertVector {
    static constexpr int NV = sizeof(VF);
    using F =  decltype(VType<VF>::elem(VF()));
    static constexpr int N = NV/sizeof(F);
    typedef typename IntType<F>::type __attribute__( ( vector_size(NV) ) ) itype;
    static VF impl(itype i) { VF f; for (int j=0;j<N;++j) f[j]=i[j]; return f;}
    static itype impl(VF f) { itype i; for (int j=0;j<N;++j) i[j]=f[j]; return i;}
  };

  // to be specialized for 2,4,8,16 float and double
  template<>
  struct ConvertVector<float32x4_t> {
    using VF = float32x4_t;
    static constexpr int NV = sizeof(VF);
    using F =  decltype(VType<VF>::elem(VF()));
    static constexpr int N = NV/sizeof(F);
    typedef typename IntType<F>::type __attribute__( ( vector_size(NV) ) ) itype;
    static VF impl(itype i) { return VF(_mm_cvtepi32_ps(__m128i(i)));}
    static itype impl(VF f) { return itype(_mm_cvttps_epi32(__m128(f)));}
  };


inline
bool testz(int32x4_t const t) {
   constexpr int32x4_t mask = {~0,~0,~0,~0};
   return _mm_testz_si128(__m128i(t),__m128i(mask)); 
}


  
#ifdef __AVX__
  template<>
  struct ConvertVector<float32x8_t> {
    using VF = float32x8_t;
    static constexpr int NV = sizeof(VF);
    using F =  decltype(VType<VF>::elem(VF()));
    static constexpr int N = NV/sizeof(F);
    typedef typename IntType<F>::type __attribute__( ( vector_size(NV) ) ) itype;
    static VF impl(itype i) { return VF(_mm256_cvtepi32_ps(__m256i(i)));}
    static itype impl(VF f) { return itype(_mm256_cvtps_epi32(__m256(f)));}
  };

  inline
  bool testz(int32x8_t const t) {
   constexpr int32x8_t mask = {~0,~0,~0,~0,~0,~0,~0,~0}; 
   return _mm256_testz_si256(__m256i(t),__m256i(mask));
  }

#endif
  


  
  template<typename VF>
  struct toIF {
    // VF is a float type vect
    // F is the float type
    static constexpr int NV = sizeof(VF);
    using F =  decltype(VType<VF>::elem(VF()));
    static constexpr int N = NV/sizeof(F);
    typedef typename IntType<F>::type __attribute__( ( vector_size(NV) ) ) itype;
    typedef typename UIntType<F>::type __attribute__( ( vector_size(NV) ) ) uitype;
    static itype ftoi(VF f) { itype i; memcpy(&i,&f,NV); return i;}
    static VF itof(itype i) { VF f; memcpy(&f,&i,NV); return f;}
    static uitype ftoui(VF f) { uitype i; memcpy(&i,&f,NV); return i;}
    static VF uitof(uitype i) { VF f; memcpy(&f,&i,NV); return f;}

    static VF convert(itype i) { return ConvertVector<VF>::impl(i);}
    static itype convert(VF f) { return ConvertVector<VF>::impl(f);}
  };

  template<>
  struct toIF<float> {
    using VF = float;
    static constexpr int NV = sizeof(VF);
    using itype = int;
    using uitype = unsigned int;
    static itype ftoi(VF f) { itype i; memcpy(&i,&f,NV); return i;}
    static VF itof(itype i) { VF f; memcpy(&f,&i,NV); return f;}
    static uitype ftoui(VF f) { uitype i; memcpy(&i,&f,NV); return i;}
    static VF uitof(uitype i) { VF f; memcpy(&f,&i,NV); return f;}

    static VF convert(itype i) { return i;}
    static itype convert(VF f) { return f;}
  };

  
  template<typename V1>
  V1 max(V1 a, V1 b) {
    return (a>b) ? a : b;
  }
  template<typename V1>
  V1 min(V1 a, V1 b) {
    return (a<b) ? a : b;
  }
  
  template<typename V1>
  V1 abs(V1 a) {
    return (a>0) ? a : -a;
  }



#ifdef SCALAR
  constexpr unsigned int VSIZE = 1;
  using FVect = float;
  constexpr FVect vzero={0};
#else
#ifdef __AVX__
  constexpr unsigned int VSIZE = 8;
  using FVect = float32x8_t;
  constexpr FVect vzero{0,0,0,0,0,0,0,0};
#else
  constexpr unsigned int VSIZE = 4;
  using FVect = float32x4_t;
  constexpr FVect vzero{0,0,0,0};
#endif
#endif  // scalar

  using IVect = ConvertVector<FVect>::itype;
  IVect convert(FVect v) { return toIF<FVect>::convert(v);}
  FVect convert(IVect v) { return toIF<FVect>::convert(v);}
  
} // nativeVector

#include<iostream>
inline
std::ostream& operator<<(std::ostream& co, nativeVector::FVect const& v) {
  co << '('<< v[0];
  for (unsigned int i=1;  i<nativeVector::VSIZE; ++i) co <<','<<v[i];
  return co <<')';
}
inline
std::ostream& operator<<(std::ostream& co, nativeVector::IVect const& v) {
  co << '('<< v[0];
  for (unsigned int i=1;  i<nativeVector::VSIZE; ++i) co <<','<<v[i];
  return co <<')';
}

#endif // NATIVE_VECTOR_H
