#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cstring>
#include <array>


#include "vdtMath.h"


#include <x86intrin.h>

typedef unsigned long long __attribute__( ( vector_size( 32 ) ) ) uint64x4_t;
typedef signed long long __attribute__( ( vector_size( 32 ) ) ) int64x4_t;
typedef unsigned int __attribute__( ( vector_size( 32 ) ) ) uint32x8_t;
typedef int __attribute__( ( vector_size( 32 ) ) ) int32x8_t;
typedef float __attribute__( ( vector_size( 32 ) ) ) float32x8_t;

typedef unsigned long long __attribute__( ( vector_size( 16 ) ) ) uint64x2_t;
typedef signed long long __attribute__( ( vector_size( 16 ) ) ) int64x2_t;
typedef signed int __attribute__( ( vector_size( 16 ) ) ) int32x4_t;
typedef unsigned int __attribute__( ( vector_size( 16 ) ) ) uint32x4_t;
typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;


#ifndef APPROX_MATH_N
#define APPROX_MATH_N
namespace approx_math {
  union binary32 {
    binary32() : ui32(0) {};
    binary32(float ff) : f(ff) {};
    binary32(int32_t ii) : i32(ii){}
    binary32(uint32_t ui) : ui32(ui){}
    
    uint32_t ui32; /* unsigned int */                
    int32_t i32; /* Signed int */                
    float f;
  };

  union binary64 {
    binary64() : ui64(0) {};
    binary64(double ff) : f(ff) {};
    binary64(int64_t ii) : i64(ii){}
    binary64(uint64_t ui) : ui64(ui){}
    
    uint64_t ui64; /* unsigned int */                
    int64_t i64; /* Signed int */                
    double f;
  };


  union binary128 {
    binary128() : ul{0,0} {};
    binary128(float32x4_t ff) : f(ff) {};
    binary128(int32x4_t ii) : i(ii){}
    binary128(uint32x4_t ii) : ui(ii){}
    binary128(int64x2_t ii) : l(ii){}
    binary128(uint64x2_t ii) : ul(ii){}
    
    __m128i i128;
    float32x4_t f;
    int32x4_t i;
    uint32x4_t ui;
    int64x2_t l;
    uint64x2_t ul;
  };

#ifdef __AVX2__
  union binary256 {
    binary256() : ul{0,0} {};
    binary256(float32x8_t ff) : f(ff) {};
    binary256(int32x8_t ii) : i(ii){}
    binary256(uint32x8_t ii) : ui(ii){}
    binary256(int64x4_t ii) : l(ii){}
    binary256(uint64x4_t ii) : ul(ii){}
    
    __m256i i256;
    float32x8_t f;
    int32x8_t i;
    uint32x8_t ui;
    int64x4_t l;
    uint64x4_t ul;
  };
#endif


}
#endif


#include "approx_log.h"

inline
void frex(float x, int & er, float & mr) {
  using namespace approx_math;

  binary32 xx,m;
  xx.f = x;
  
  // as many integer computations as possible, most are 1-cycle only, and lots of ILP.
  int e= (((xx.i32) >> 23) & 0xFF) -127; // extract exponent
  m.i32 = (xx.i32 & 0x007FFFFF) | 0x3F800000; // extract mantissa as an FP number
  
  int adjust = (xx.i32>>22)&1; // first bit of the mantissa, tells us if 1.m > 1.5
  m.i32 -= adjust << 23; // if so, divide 1.m by 2 (exact operation, no rounding)
  e += adjust;           // and update exponent so we still have x=2^E*y
  
  er = e;
  // now back to floating-point
  mr = m.f; // 
  // all the computations so far were free of rounding errors...
}

inline
void frex(double x, int & er, double & mr) {
  using namespace approx_math;

  binary64 xx,m;
  xx.f = x;
  
  // as many integer computations as possible, most are 1-cycle only, and lots of ILP.
  int e=  int( ( xx.ui64 >> 52) & 0x7FF) -1023; // extract exponent
  m.ui64 = (xx.ui64 & 0x800FFFFFFFFFFFFFULL) | 0x3FF0000000000000ULL; // extract mantissa as an FP number
  
  long long adjust = (xx.ui64>>51)&1; // first bit of the mantissa, tells us if 1.m > 1.5
  m.i64 -= adjust << 52; // if so, divide 1.m by 2 (exact operation, no rounding)
  e += adjust;           // and update exponent so we still have x=2^E*y
  
  er = e;
  // now back to floating-point
  mr = m.f; // 
  // all the computations so far were free of rounding errors...
}



inline
void irex(float x, int & er, unsigned int & mr) {
  using namespace approx_math;

  binary32 xx,m;
  xx.f = x;
  
  // as many integer computations as possible, most are 1-cycle only, and lots of ILP.
  er = (((xx.ui32) >> 23) & 0xFF) -127; // extract exponent
  mr = (xx.ui32 & 0x007FFFFF) | 0x00800000; // extract mantissa as an integer number

}

inline
unsigned int mult(unsigned int a, unsigned int b) {
  constexpr unsigned int Q = 23;
  constexpr unsigned long long K  = (1 << (Q-1));
  unsigned long long  temp = (unsigned long long)(a) * (unsigned long long)(b); // result type is operand's type
  // Rounding; mid values are rounded up
  temp += K;
  // Correct by dividing by base
  return (temp >> Q);  
}


struct IExMa {
  IExMa(float r=1.f) { irex(r,er,mr); }

  int er=0; 
  unsigned int mr=1;
  
#pragma omp declare simd
  IExMa & operator+=(IExMa b) {
    auto k = mult(mr,b.mr);
    unsigned int ov = k >> 31;  mr = k>>ov;
    er += b.er+ov;
    return *this;
  }

#pragma omp declare simd
  IExMa & reduce(IExMa b) {
    mr = mult(mr>>4,b.mr>>4);
    er += b.er+8;
    return *this;
  }
};

#pragma omp declare reduction (foo:struct IExMa: omp_out.reduce(omp_in))



inline
void irex(double x, int & er, unsigned long long & mr) {
  using namespace approx_math;

  binary64 xx;
  xx.f = x;
  
  // as many integer computations as possible, most are 1-cycle only, and lots of ILP.
  er =  int( ( xx.ui64 >> 52) & 0x7FF) -1023; // extract exponent
  mr = (xx.ui64 & 0x000FFFFFFFFFFFFFULL) | 0x0010000000000000ULL; // extract mantissa as an integer number

}

inline
__uint128_t multD(__uint128_t a, __uint128_t b) {
  constexpr int Q = 52;
  constexpr unsigned long long K  = (1UL << (Q-1));
  auto temp = a*b;
  temp += K;
  return temp >> Q;
}




/*   a waiste
uint64x4_t mult(uint64x4_t a,uint64x4_t b) {
  constexpr int Q = 23;
  constexpr unsigned long long K  = (1 << (Q-1));
  uint64x4_t temp = a*b;
  temp += K;
  return temp >> Q;
}
*/

/* avx2...
uint64x4_t multI(uint64x4_t a,uint64x4_t b) {
  constexpr int Q = 23;
  constexpr unsigned long long K  = (1 << (Q-1));
  uint64x4_t temp =  uint64x4_t(_mm256_mul_epu32(__m256i(a),__m256i(b)));
  temp += K;
  return temp >> Q;
}
*/


inline
void irex(float32x4_t x, int32x4_t & er, uint32x4_t & mr) {
  using namespace approx_math;

  binary128 xx;
  xx.f = x;
  
  // as many integer computations as possible, most are 1-cycle only, and lots of ILP.
  er = (((xx.i) >> 23) & 0xFF) -127; // extract exponent
  mr = (xx.ui & 0x007FFFFF) | 0x00800000; // extract mantissa as an integer number

}

#ifdef __AVX2__
inline
void irex(float32x8_t x, int32x8_t & er, uint32x8_t & mr) {
  using namespace approx_math;

  binary256 xx;
  xx.f = x;
  
  // as many integer computations as possible, most are 1-cycle only, and lots of ILP.
  er = (((xx.i) >> 23) & 0xFF) -127; // extract exponent
  mr = (xx.ui & 0x007FFFFF) | 0x00800000; // extract mantissa as an integer number

}
#endif

#include<iostream>
 
inline
uint32x4_t multI(uint32x4_t aa, uint32x4_t bb) {
  using namespace approx_math;
  binary128 a(aa), b(bb);
  binary128 temp1, temp2;
  constexpr int Q = 23;
  constexpr unsigned long long K  = (1 << (Q-1));
  temp1.i128 = _mm_mul_epu32(a.i128,b.i128);
  temp1.ul += K;  temp1.ul >>= Q; // results are in position 0 and 2

  a.ul >>= 32; b.ul >>= 32;  // move "odd" integers in position 
  /*  same speed on SB!
  constexpr int32x4_t mask{1,0,3,2};
  a.ui = __builtin_shuffle(a.ui,mask);
  b.ui = __builtin_shuffle(b.ui,mask);
  */
  temp2.i128 = _mm_mul_epu32(a.i128,b.i128);
  temp2.ul += K; temp2.ul >>= Q;
   //  std::cout << "temp " << temp1.ul[0] << " " << temp1.ul[1] << " " << temp2.ul[0] << " " << temp2.ul[1] << std::endl;
  // std::cout << "temp " << temp1.ui[0] << " " << temp1.ui[2] << " " << temp2.ui[0] << " " << temp2.ui[2] << std::endl;  // BHO
  
  temp2.ul <<= 32;  // results are now in position 1 and 3
  temp1.ul |=temp2.ul;  

 // constexpr int32x4_t mask2{0,4,2,6};
 // temp1.i = __builtin_shuffle(temp1.i,temp2.i,mask2);
  // std::cout << "temp " << temp1.ui[0] << " " << temp1.ui[1] << " " << temp1.ui[2] << " " << temp1.ui[3] << std::endl;
  return temp1.ui;
}

inline
void multI2(uint64x2_t & aa1, uint64x2_t & aa2, uint32x4_t bb) {
  using namespace approx_math;
  binary128 a1(aa1), a2(aa2), b(bb);
  constexpr int Q = 23;
  constexpr unsigned long long K  = (1 << (Q-1));
  a1.i128 = _mm_mul_epu32(a1.i128,b.i128);
  a1.ul += K;  a1.ul >>= Q; 

  b.ul >>= 32;  // move "odd" integers in "even" positions 

  a2.i128 = _mm_mul_epu32(a2.i128,b.i128);
  a2.ul += K; a2.ul >>= Q;
  aa1 = a1.ul; aa2=a2.ul;
}

#ifdef __AVX2__
inline
void multI2(uint64x4_t & aa1, uint64x4_t & aa2, uint32x8_t bb) {
  using namespace approx_math;
  binary256 a1(aa1), a2(aa2), b(bb);
  constexpr int Q = 23;
  constexpr unsigned long long K  = (1 << (Q-1));
  a1.i256 = _mm256_mul_epu32(a1.i256,b.i256);
  a1.ul += K;  a1.ul >>= Q; 

  b.ul >>= 32;  // move "odd" integers in "even" positions 

  a2.i256 = _mm256_mul_epu32(a2.i256,b.i256);
  a2.ul += K; a2.ul >>= Q;
  aa1 = a1.ul; aa2=a2.ul;
}
#endif



#include<random>
#include<iostream>
#include<cstdio>

float32x4_t doMult(float32x4_t a, float32x4_t b) {
  int32x4_t ea,eb; uint32x4_t ma,mb;
  irex(a, ea,ma);
  irex(b, eb,mb);

  // std::cout << ea[0] << " " << ea[1] << " " << ea[2] << " " << ea[3] << std::endl;
  // std::cout << ma[0] << " " << ma[1] << " " << ma[2] << " " << ma[3] << std::endl;


  auto r = multI(ma,mb);

  auto ires = ea+eb-23;
  float32x4_t res;
  for (int i=0; i!=4; ++i) res[i] = log2(r[i])+ires[i]; 
  return res;
  
} 

float doMult(float a, float b) {
  int ea,eb; unsigned int ma,mb;
  irex(a, ea,ma);
  irex(b, eb,mb);

  unsigned int r = mult(ma,mb);

  return ea+eb+log2(r)-23;

} 

float doMultD(double a, double b) {
  int ea,eb; unsigned long long ma,mb;
  irex(a, ea,ma);
  irex(b, eb,mb);

  unsigned long long r = multD(ma,mb);

  return ea+eb+log2(r)-52;

} 


unsigned int taux=0;
inline volatile unsigned long long rdtsc() {
 return __rdtscp(&taux);
}




int main(int argc, char**) {
  std::mt19937 eng;
  std::mt19937 eng2;
  std::uniform_real_distribution<float> rgen(0.,1.);
  std::uniform_real_distribution<float> rgenL(0.,0.01);
  std::uniform_real_distribution<float> rgenM(0.4,0.6);
  std::uniform_real_distribution<float> rgenH(0.99,1.0);


  std::cout << log2(.5*.7) << " " << doMult(.5f,.7f) << std::endl;
  std::cout << log2(.1*.7) << " " << doMult(.1f,.7f) << std::endl;
  std::cout << log2(.1e-5*.7) << " " << doMult(.1e-5f,.7f) << std::endl;
  std::cout << log2(.99*.99) << " " << doMult(.99f,.99f)  << " " << doMultD(.99,.99)<< std::endl;
  std::cout << log2(.999*.999) << " " << doMult(.999f,.999f)  << " " << doMultD(.999,.999)<< std::endl;
  std::cout << log2(.9999*.9999) << " " << doMult(.9999f,.9999f)  << " " << doMultD(.9999,.9999)<< std::endl;
  unsigned int big = 0x0affffff; unsigned int big1 = 0x7fffffff; unsigned int bigm = 0x00FFFFFF;
  __uint128_t mmm = __uint128_t(big)*__uint128_t(big); mmm+= (1 << 22); mmm>>=23;
  printf("%x %llx\n",mult(big,big),(unsigned long long)(mmm));
  mmm = __uint128_t(big1)*__uint128_t(bigm); mmm+= (1 << 22); mmm>>=23;
  printf("%x %llx\n",mult(big1,bigm),(unsigned long long)(mmm));

  float32x4_t a{.5f,.1f,.1e-5f,.1e-10f};
  float32x4_t b{.7f,.7f,.7f,.7e-10f};
  auto c = doMult(a,b);
  std::cout << c[0] << " " << c[1] << " " << c[2] << " " << c[3] << std::endl;


  {
    {
    int er=0; float mr=0;
    frex(0.49,er,mr); std::cout << mr << " ";
    frex(0.51,er,mr); std::cout << mr << " ";
    frex(0.74,er,mr); std::cout << mr << " ";
    frex(0.76,er,mr); std::cout << mr << " ";
    frex(0.99,er,mr); std::cout << mr << " ";
    frex(1.01,er,mr); std::cout << mr << " ";
    frex(1.49,er,mr); std::cout << mr << " ";
    frex(1.51,er,mr); std::cout << mr << " ";
    frex(1.74,er,mr); std::cout << mr << " ";
    frex(1.76,er,mr); std::cout << mr << " ";
    frex(1.99,er,mr); std::cout << mr << " ";
    frex(0.999,er,mr); std::cout << mr << " ";
    frex(0.9999,er,mr); std::cout << mr << " ";
    frex(0.99999,er,mr); std::cout << mr << std::endl;
    }
    {
    int er=0; double mr=0;
    frex(0.49,er,mr); std::cout << mr << " ";
    frex(0.51,er,mr); std::cout << mr << " ";
    frex(0.74,er,mr); std::cout << mr << " ";
    frex(0.76,er,mr); std::cout << mr << " ";
    frex(0.99,er,mr); std::cout << mr << " ";
    frex(1.01,er,mr); std::cout << mr << " ";
    frex(1.49,er,mr); std::cout << mr << " ";
    frex(1.51,er,mr); std::cout << mr << " ";
    frex(1.74,er,mr); std::cout << mr << " ";
    frex(1.76,er,mr); std::cout << mr << " ";
    frex(1.99,er,mr); std::cout << mr << " ";
    frex(0.999,er,mr); std::cout << mr << " ";
    frex(0.9999,er,mr); std::cout << mr << " ";
    frex(0.99999,er,mr); std::cout << mr << std::endl;
    }

  {
    {
    int er=0; unsigned int mr=0;
    irex(0.49,er,mr); std::cout << mr << " ";
    irex(0.51,er,mr); std::cout << mr << " ";
    irex(0.74,er,mr); std::cout << mr << " ";
    irex(0.76,er,mr); std::cout << mr << " ";
    irex(0.99,er,mr); std::cout << mr << " ";
    irex(1.01,er,mr); std::cout << mr << " ";
    irex(1.49,er,mr); std::cout << mr << " ";
    irex(1.51,er,mr); std::cout << mr << " ";
    irex(1.74,er,mr); std::cout << mr << " ";
    irex(1.76,er,mr); std::cout << mr << " ";
    irex(1.99,er,mr); std::cout << mr << " ";
    irex(0.999,er,mr); std::cout << mr << " ";
    irex(0.9999,er,mr); std::cout << mr << " ";
    irex(0.99999,er,mr); std::cout << mr << std::endl;
    }
    {
    int er=0; unsigned long long mr=0;
    irex(0.49,er,mr); std::cout << mr << " ";
    irex(0.51,er,mr); std::cout << mr << " ";
    irex(0.74,er,mr); std::cout << mr << " ";
    irex(0.76,er,mr); std::cout << mr << " ";
    irex(0.99,er,mr); std::cout << mr << " ";
    irex(1.01,er,mr); std::cout << mr << " ";
    irex(1.49,er,mr); std::cout << mr << " ";
    irex(1.51,er,mr); std::cout << mr << " ";
    irex(1.74,er,mr); std::cout << mr << " ";
    irex(1.76,er,mr); std::cout << mr << " ";
    irex(1.99,er,mr); std::cout << mr << " ";
    irex(0.999,er,mr); std::cout << mr << " ";
    irex(0.9999,er,mr); std::cout << mr << " ";
    irex(0.99999,er,mr); std::cout << mr << std::endl;
    }
  }


    float mf=1.f; 
    for (int i=0;i<129; ++i) 
      mf*=1.5f;
    std::cout << mf << std::endl;
    mf=1.f;
    for (int i=0;i<129; ++i) 
      mf*=.75f;
    std::cout << mf << std::endl;
  }

  if (argc>1) return 1;

  long long tl=0, t00=0, t0=0, t01=0, t1=0,t11=0,t12=0, t2=0, t21=0,  t22=0, t23=0, t24=0,  t3=0,  t4=0;
  double em0=0, em01=0, em1=0, em11=0,em12=0, em2=0, em21=0, em22=0, em23=0, em24=0,  em3=0,  em4=0;

  //constexpr int NN=1024;
  constexpr int NN=1000000;
  alignas(32) std::array<float, NN> r;

  constexpr int NL=100;
  //constexpr int NL=100000;

  for (int ok=1; ok!=NL+4; ++ok) {
    bool pr = ok<=argc || ok>NL-1;

    if (ok<NL+1) 
      for (int i=0;i!=NN;++i)
	r[i]=rgen(eng);
    else if (ok==NL+1) {
      std::cout << "low prob" << std::endl;
      for (int i=0;i!=NN;++i)
	r[i]=rgenL(eng);
    } else if (ok==NL+2) {
      std::cout << "med prob" << std::endl;
      for (int i=0;i!=NN;++i)
	r[i]=rgenM(eng);
    } else {
      std::cout << "high prob" << std::endl;
      for (int i=0;i!=NN;++i)
	r[i]=rgenH(eng);
    }
   
    // reference loop
    tl -= rdtsc();
    float sq=0.;
    for (int i=0;i!=NN;++i)
      sq+= r[i];
    tl += rdtsc();

   if(pr)
     printf("sum %f : %f\n",double(tl)/double(NN*ok),sq);

#ifndef NOSTD
    // double precision std::log2 (accumulation in 128)
    t00 -= rdtsc();
    double sl=0.;
    __float128 ss=0.;
    for (int i=0;i!=NN;++i)
      ss+= ::log2(double(r[i]));
    t00 += rdtsc();
    sl = ss;

    // add inverse, if correct should be zero!)
    auto zl=sl;
    for (int i=0;i!=NN;++i)
      zl+= ::log2(1./double(r[i]));

    if(pr)
      printf("float128 %f : %f %f %a\n",double(t00)/double(NN*ok),sl,zl,sl);

   {
      // double precision std::log2
      t1 -= rdtsc();
      double sv=0.;
      for (int i=0;i!=NN;++i)
	sv+= ::log2(double(r[i]));
      t1 += rdtsc();
      
      // add inverse, if correct should be zero!)
      double zv=sv;
      for (int i=0;i!=NN;++i)
	zv+= ::log2(1./double(r[i]));
    
      em1 = std::max(em1, -std::abs(sv-sl)/sl);  
      if(pr)
	printf("sdt %f : %f %f %a %e %e\n",double(t1)/double(NN*ok),sv ,zv, sv, (sv-sl)/sl,em1);
      
    }
#else
   double sl = 1.;
#endif



    {
      // double precision vdt::log2
      t0 -= rdtsc();
      double sv=0.;
      for (int i=0;i!=NN;++i)
	sv+= vdt::fast_log(double(r[i]));
      t0 += rdtsc();
      
      // add inverse, if correct should be zero!)
      double zv=sv;
      for (int i=0;i!=NN;++i)
	zv+= vdt::fast_log(1./double(r[i]));
    
      em0 = std::max(em0, -std::abs(sv/0.693147182464599609375-sl)/sl);  
      if(pr)
	printf("vdt %f : %f %f %a %e %e\n",double(t0)/double(NN*ok),sv/0.693147182464599609375 ,zv,sv/0.693147182464599609375, (sv/0.693147182464599609375-sl)/sl,em0);
      
    }

    {
      // single preciosn  approxmath
      t01 -= rdtsc();
      double sv=0.;
      for (int i=0;i!=NN;++i)
	sv+= unsafe_logf<8>(r[i]);
      t01 += rdtsc();
      
      // add inverse, if correct should be zero!)
      double zv=sv;
      for (int i=0;i!=NN;++i)
	zv+= unsafe_logf<8>(1./r[i]);
    
      em01 = std::max(em01, -std::abs(sv/0.693147182464599609375-sl)/sl);  
      if(pr)
	printf("approx %f : %f %f %a %e %e\n",double(t01)/double(NN*ok),sv/0.693147182464599609375 ,zv,sv/0.693147182464599609375, (sv/0.693147182464599609375-sl)/sl,em01);
      
    }

#ifndef	NOSTD
    {
      // frexp : 
      t11 -= rdtsc();
      int sf=0; double mf=1; 
      for (int i=0;i!=NN; ++i) {
	int er=0;
	double mr = ::frexp(r[i],&er);
	sf+=er; mf*=mr;
        mf = ::frexp(mf,&er); sf+=er;
      }
      t11 += rdtsc();
      
      em11 = std::max(em11, -std::abs(sf+std::log2(mf)-sl)/sl);
      if(pr)
	printf("frexp %f : %d %f %f %a %e %e\n",double(t11)/double(NN*ok),sf,mf,sf+std::log2(mf),sf+std::log2(mf), (sf+std::log2(mf)-sl)/sl,em11);
      
      
      // add inverse, if correct should be zero!)
      auto zf=sf;
      for (int i=0;i<NN; ++i) {
	int er=0;
	double mr = ::frexp(1./r[i],&er);
	zf+=er;
	mf = ::frexp(mf*mr,&er); zf+=er;
      }
      
      if(pr)
	printf("zero fr : %d %f %f %a\n",zf,mf,zf+std::log2(mf),zf+std::log2(mf));
    }


    {
      // frexp : the second frexp on the sum will not vectorize. mantissas are small, we can multiply 128 of them (512 in double!  (double accumulation)
      t12 -= rdtsc();
      int sf=0; double mf=1.; 
      for (int i=0;i<NN; i+=512) {
	double mi=1.f; 
	for (auto k=i; k!=std::min(i+512,NN); ++k) {
	  int er=0;
	  double mr = ::frexp(r[k],&er);
	  sf+=er; mi*=mr;
	}
	int ei=0; mf = frexp(mf*mi,&ei); sf+=ei;
      }
      t12 += rdtsc();
      
      em12 = std::max(em12, -std::abs(sf+std::log2(mf)-sl)/sl);

      if(pr)
	printf("frexpv %f : %d %f %f %a %e %e\n",double(t12)/double(NN*ok),sf,mf,sf+std::log2(mf),sf+std::log2(mf), (sf+std::log2(mf)-sl)/sl,em12);
      
      
      // add inverse, if correct should be zero!)
      auto zf=sf;
      for (int i=0;i<NN; i+=512) {
	double mi=1.f; 
	for (auto k=i; k!=std::min(i+512,NN); ++k) {
	  int er=0;
	  double mr = ::frexp(1.f/r[k],&er);
	  zf+=er; mi*=mr;
	}
	int ei=0; mf = ::frexp(mf*mi,&ei); zf+=ei;
      }
      
      if(pr)
	printf("zero fv : %d %f %f %a\n",zf,mf,zf+std::log2(mf),zf+std::log2(mf));
    }
#endif


    { 
      // full integer (will never vectorize!)  not even  with openmp4 (no magic...)!
      t2 -= rdtsc();
      IExMa isp;
#pragma omp simd aligned(r : 32) reduction(foo:isp)
      for (int i=0;i<NN;++i) {
	IExMa l(r[i]);
	isp+=l;
      }
      t2 += rdtsc();
      auto si = isp.er;  auto pi = isp.mr;
      em2 = std::max(em2, -std::abs(si+std::log2(pi)-23-sl)/sl);

      if(pr)
	printf("int %f : %d %d %f %a %e %e\n",double(t2)/double(NN*ok),si,pi,si+std::log2(pi)-23,si+std::log2(pi)-23, (si+std::log2(pi)-23-sl)/sl,em2);
      
      // add inverse, if correct should be zero!)
      int zi=si;
      for (int i=0;i!=NN;++i) {
	int er=0; unsigned int mr=0;
	irex(1./r[i],er,mr);
	zi+=er; pi=mult(pi,mr);
	//      if (pi > 0x00800000) { pi/=2; zi++;}
	if (pi >= 0x80000000) { pi/=2; zi++;}
      }
 
      if(pr)
	printf("zero : %d %d %f %a\n",zi,pi,zi+std::log2(pi)-23,zi+std::log2(pi)-23);


    }
    {

      // full integer (unrolled...)
      t21 -= rdtsc();
      int si=0; unsigned int ipi=0;
      irex(1.f,si,ipi); 
      unsigned int pi[4]={ipi,ipi,ipi,ipi};
      int sk[4]={si,si,si,si};
      for (int i=0;i<NN;i+=4) {
	for (int k=0; k!=4; ++k) {
	  int er=0; unsigned int mr=0;
	  irex(r[i+k],er,mr);
	  sk[k]+=er; pi[k]=mult(pi[k],mr);
	  unsigned int ov = pi[k] >> 31;  pi[k]>>=ov; sk[k]+=ov; // avoid overflow
	}
      }
      si = sk[0]+sk[1]+sk[2]+sk[3]+8+16;
      ipi = mult(mult(pi[0]>>4,pi[1]>>4)>>4,mult(pi[2]>>4,pi[3]>>4)>>4);
      t21 += rdtsc();
      
      em21 = std::max(em21, -std::abs(si+std::log2(ipi)-23-sl)/sl);
     if(pr)
       printf("intu %f : %d %d %f %a %e %e\n",double(t21)/double(NN*ok),si,ipi,si+std::log2(ipi)-23,si+std::log2(ipi)-23, (si+std::log2(ipi)-23-sl)/sl,em21);

    }

    {

      // full long long (unrolled...)
      t24 -= rdtsc();
      int si=0; unsigned long long ipi=0;
      irex(1.,si,ipi); 
      __uint128_t pi[4]={ipi,ipi,ipi,ipi};
      int sk[4]={si,si,si,si};
      for (int i=0;i<NN;i+=4) {
	for (int k=0; k!=4; ++k) {
	  int er=0; unsigned long long mr=0;
	  irex(double(r[i+k]),er,mr);
	  sk[k]+=er; pi[k]=multD(pi[k],mr);
	  unsigned long long ov = pi[k] >> 63;  pi[k]>>=ov; sk[k]+=ov; // avoid overflow
	}
      }
      constexpr int shift=8;
      si = sk[0]+sk[1]+sk[2]+sk[3]+ 6*shift;
      ipi = multD(multD(pi[0]>>shift,pi[1]>>shift)>>shift,multD(pi[2]>>shift,pi[3]>>shift)>>shift);
      t24 += rdtsc();
      
      em24 = std::max(em24, -std::abs(si+std::log2(ipi)-52-sl)/sl);
     if(pr)
       printf("longu %f : %d %d %f %a %e %e\n",double(t24)/double(NN*ok),si,ipi,si+std::log2(ipi)-52,si+std::log2(ipi)-52, (si+std::log2(ipi)-52-sl)/sl,em24);

    }



    {
      // full integer (vectorized???)
      t22 -= rdtsc();
      int si=0; unsigned int ipi=0;
      irex(1.f,si,ipi); 
      // uint32x4_t pi = {ipi,ipi,ipi,ipi};
      uint64x2_t p1 = {ipi,ipi};
      uint64x2_t p2 = {ipi,ipi};
      uint64x2_t s1{0,0};
      int32x4_t sk = {si,si,si,si};
      for (int i=0;i<NN;i+=8) {
	for (int j = i; j<i+8; j+=4)
	{
	  int32x4_t er; uint32x4_t mi; 
	  float32x4_t ri{r[j+0],r[j+1],r[j+2],r[j+3]};
	  irex(ri,er,mi);  sk+=er;
	  multI2(p1,p2,mi);
	  approx_math::binary128 ov1(p1 >> 31);  p1>>=ov1.ul; s1+=ov1.ul; // avoid overflow
	  approx_math::binary128 ov2(p2 >> 31);  p2>>=ov2.ul; s1+=ov2.ul; // avoid overflow
	}
      }
      si = sk[0]+sk[1]+sk[2]+sk[3]+8+16; si+=s1[0]+s1[1];
      ipi = mult(mult(p1[0]>>4,p1[1]>>4)>>4,mult(p2[0]>>4,p2[1]>>4)>>4);
 
      t22 += rdtsc();
      
      em22 = std::max(em22, -std::abs(si+std::log2(ipi)-23-sl)/sl);

      if(pr)
	printf("intv %f : %d %d %f %a %e %e\n",double(t22)/double(NN*ok),si,ipi,si+std::log2(ipi)-23,si+std::log2(ipi)-23, (si+std::log2(ipi)-23-sl)/sl,em22);

    }


#ifdef __AVX2__
    {
      // full integer ( avx2 vectorized???)
      t23 -= rdtsc();
      int si=0; unsigned int ipi=0;
      irex(1.f,si,ipi); 
      // uint32x4_t pi = {ipi,ipi,ipi,ipi};
      uint64x4_t p1 = {ipi,ipi,ipi,ipi};
      uint64x4_t p2 = {ipi,ipi,ipi,ipi};
      uint64x4_t s1{0,0,0,0};
      int32x8_t sk = {si,si,si,si,si,si,si,si};
      for (int i=0;i<NN;i+=16) {
	for (int j = i; j<i+16; j+=8)
	{
	  int32x8_t er; uint32x8_t mi; 
	  float32x8_t ri{r[j+0],r[j+1],r[j+2],r[j+3],r[j+4],r[j+5],r[j+6],r[j+7]};
	  irex(ri,er,mi);  sk+=er;
	  multI2(p1,p2,mi);
	  approx_math::binary256 ov1(p1 >> 31);  p1>>=ov1.ul; s1+=ov1.ul; // avoid overflow
	  approx_math::binary256 ov2(p2 >> 31);  p2>>=ov2.ul; s1+=ov2.ul; // avoid overflow
	}
      }
      si = sk[0]+sk[1]+sk[2]+sk[3]+sk[4]+sk[5]+sk[6]+sk[7] + 14*4; si+=s1[0]+s1[1]+s1[2]+s1[3];
      ipi = mult( 
		 mult(mult(p1[0]>>4,p1[1]>>4)>>4,mult(p2[0]>>4,p2[1]>>4)>>4)>>4,
		 mult(mult(p1[2]>>4,p1[3]>>4)>>4,mult(p2[2]>>4,p2[3]>>4)>>4)>>4
		  );
 
      t23 += rdtsc();
      
      em23 = std::max(em23, -std::abs(si+std::log2(ipi)-23-sl)/sl);

      if(pr)
	printf("intv2 %f : %d %d %f %a %e %e\n",double(t23)/double(NN*ok),si,ipi,si+std::log2(ipi)-23,si+std::log2(ipi)-23, (si+std::log2(ipi)-23-sl)/sl,em23);

    }

#endif
 


    {
    // frexp like: the second frexp on the sum will not vectorize. mantissas are small, we can multiply 128 of them!
    t3 -= rdtsc();
    int sf=0; float mf=1.f; 
    for (int i=0;i<NN; i+=128) {
      float mi=1.f; 
      for (auto k=i; k!=std::min(i+128,NN); ++k) {
	int er=0; float mr=0;
	frex(r[k],er,mr);
	sf+=er; mi*=mr;
      }
      int ei=0; frex(mf*mi,ei,mf); sf+=ei;
    }
    t3 += rdtsc();


    em3 = std::max(em3, -std::abs(sf+std::log2(mf)-sl)/sl);

    if(pr)
      printf("intf %f : %d %f %f %a %e %e\n",double(t3)/double(NN*ok),sf,mf,sf+std::log2(mf),sf+std::log2(mf), (sf+std::log2(mf)-sl)/sl,em3);

   
    // add inverse, if correct should be zero!)
    auto zf=sf;
    for (int i=0;i<NN; i+=128) {
      float mi=1.f; 
      for (auto k=i; k!=std::min(i+128,NN); ++k) {
	int er=0; float mr=0;
	frex(1.f/r[k],er,mr);
	zf+=er; mi*=mr;
      }
      int ei=0; frex(mf*mi,ei,mf); zf+=ei;
    }


 
    if(pr)
      printf("zerof : %d %f %f %a\n",zf,mf,zf+std::log2(mf),zf+std::log2(mf));

    }

    {
      // frexp like: the second frexp on the sum will not vectorize. mantissas are small, we can multiply 128 of them (512 in double!  (double accumulation)
    t4 -= rdtsc();
    int sf=0; double mf=1.; 
    for (int i=0;i<NN; i+=512) {
      double mi=1.f; 
      for (auto k=i; k!=std::min(i+512,NN); ++k) {
	int er=0; float mr=0;
	frex(r[k],er,mr);
	sf+=er; mi*=mr;
      }
      int ei=0; frex(mf*mi,ei,mf); sf+=ei;
    }
    t4 += rdtsc();

    em4 = std::max(em4, -std::abs(sf+std::log2(mf)-sl)/sl);

    if(pr)
      printf("intd %f : %d %f %f %a %e %e\n",double(t4)/double(NN*ok),sf,mf,sf+std::log2(mf),sf+std::log2(mf), (sf+std::log2(mf)-sl)/sl,em4);

   
    // add inverse, if correct should be zero!)
    auto zf=sf;
    for (int i=0;i<NN; i+=512) {
      double mi=1.f; 
      for (auto k=i; k!=std::min(i+512,NN); ++k) {
	int er=0; float mr=0;
	frex(1.f/r[k],er,mr);
	zf+=er; mi*=mr;
      }
      int ei=0; frex(mf*mi,ei,mf); zf+=ei;
    }
 
    if(pr)
      printf("zerod : %d %f %f %a\n\n",zf,mf,zf+std::log2(mf),zf+std::log2(mf));
    }
  }

  return 0;
}
