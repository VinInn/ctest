#ifndef APPROX_EXP_H
#define APPROX_EXP_H
/*  Quick and not that dirty vectorizable exp implementations
    Author: Florent de Dinechin, Aric, ENS-Lyon 
    with advice from Vincenzo Innocente, CERN
    All right reserved

Warning + disclaimers:
 
Feel free to distribute or insert in other programs etc, as long as this notice is attached.
    Comments, requests etc: Florent.de.Dinechin@ens-lyon.fr

Polynomials were obtained using Sollya scripts (in comments): 
please also keep these comments attached to the code. 

If a derivative of this code ends up in the glibc I am too happy: the version with MANAGE_SUBNORMALS=1 and DEGREE=6 is faithful-accurate over the full 2^32 binary32 numbers and behaves well WRT exceptional cases. It is about 4 times faster than the stock expf on this PC, when compiled with gcc -O2.

This code is FMA-safe (i.e. accelerated and more accurate with an FMA) as long as my parentheses are respected. 

A remaining TODO is to try and manage the over/underflow using only integer tests as per Astasiev et al, RNC conf.
Not sure it makes that much sense in the vector context.

*/

// #define MANAGE_SUBNORMALS 1 // No measurable perf difference, so let's be clean.
// If set to zero we flush to zero the subnormal outputs, ie for x<-88 or so

// DEGREE 
// 6 is perfect. 
// 5 provides max 2-ulp error, 
// 4 loses 44 ulps (6 bits) for an acceleration of 10% WRT 6
// (I don't subtract the loop and call overhead, so it would be more for inlined code)

// see the comments in the code for the accuracy you get from a given degree




#include "nativeVector.h"


template<typename Float, int DEGREE>
struct approx_expf_P {  static Float impl(Float p); };

// degree =  2   => absolute accuracy is  8 bits
template<typename Float>
struct approx_expf_P<Float, 2> {
  static Float impl(Float y) {
    return   float(0x2.p0) + y * (float(0x2.07b99p0) + y * float(0x1.025b84p0)) ;
  }
};
// degree =  3   => absolute accuracy is  12 bits
template<typename Float>
struct approx_expf_P<Float, 3> {
  static Float impl(Float y) {
#ifdef HORNER  // HORNER 
    return   float(0x2.p0) + y * (float(0x1.fff798p0) + y * (float(0x1.02249p0) + y * float(0x5.62042p-4))) ;
#else // ESTRIN
    Float p23 = (float(0x1.02249p0) + y * float(0x5.62042p-4)) ;
    Float p01 = float(0x2.p0) + y * float(0x1.fff798p0);
    return p01 + y*y*p23;
#endif
  }
};
// degree =  4   => absolute accuracy is  17 bits
template<typename Float>
struct approx_expf_P<Float, 4> {
  static Float impl(Float y) {
    return   float(0x2.p0) + y * (float(0x1.fffb1p0) + y * (float(0xf.ffe84p-4) + y * (float(0x5.5f9c1p-4) + y * float(0x1.57755p-4)))) ;
  }
};
// degree =  5   => absolute accuracy is  22 bits
template<typename Float>
struct approx_expf_P<Float, 5> {
  static Float impl(Float y) {
    return   float(0x2.p0) + y * (float(0x2.p0) + y * (float(0xf.ffed8p-4) + y * (float(0x5.5551cp-4) + y * (float(0x1.5740d8p-4) + y * float(0x4.49368p-8))))) ;
  }
};
// degree =  6   => absolute accuracy is  27 bits
template<typename Float>
struct approx_expf_P<Float, 6> {
  static Float impl(Float y) {
#ifdef HORNER  // HORNER 
    float p =  float(0x2.p0) + y * (float(0x2.p0) + y * (float(0x1.p0) + y * (float(0x5.55523p-4) + y * (float(0x1.5554dcp-4) + y * (float(0x4.48f41p-8) + y * float(0xb.6ad4p-12)))))) ;
#else // ESTRIN does seem to save a cycle or two
    Float p56 = float(0x4.48f41p-8) + y * float(0xb.6ad4p-12);
    Float p34 = float(0x5.55523p-4) + y * float(0x1.5554dcp-4);
    Float y2 = y*y;
    Float p12 = float(0x2.p0) + y; // By chance we save one operation here! Funny.
    Float p36 = p34 + y2*p56;
    Float p16 = p12 + y2*p36;
    Float p =  float(0x2.p0) + y*p16;
#endif
    return p;
  }
};
// degree =  7  => absolute accuracy is  31 bits
template<typename Float>
struct approx_expf_P<Float, 7> {
  static Float impl(Float y) {
    return float(0x2.p0) + y * (float(0x2.p0) + y * (float(0x1.p0) + y * (float(0x5.55555p-4) + y * (float(0x1.5554e4p-4) + y * (float(0x4.444adp-8) + y * (float(0xb.6a8a6p-12) + y * float(0x1.9ec814p-12))))))) ;
  }
};
/* The Sollya script that computes the polynomials above


f= 2*exp(y);
I=[-log(2)/2;log(2)/2];
filename="/tmp/polynomials";
print("") > filename;
for deg from 2 to 8 do begin
  p = fpminimax(f, deg,[|1,23...|],I, Floating, absolute); 
  display=decimal;
  acc=floor(-log2(sup(supnorm(p, f, I, absolute, 2^(-40)))));
  print( "   // degree = ", deg, 
         "  => absolute accuracy is ",  acc, "bits" ) >> filename;
  print("#if ( DEGREE ==", deg, ")") >> filename;
  display=hexadecimal;
  print("   Float p = ", horner(p) , ";") >> filename;
  print("#endif") >> filename;
end;

*/

template<typename Float, bool FMA=true>
struct RangeRedution {
  static Float impl(Float x, Float z) { 
    // FMA specific range reduction
    constexpr float log2F = 0xb.17218p-4;
     return x-z*log2F;
  }
};

template<typename Float>
struct RangeRedution<Float,false> {
  static Float impl(Float x, Float z) { 
    constexpr float log2H = float(0xb.172p-4);
    constexpr float log2L = float(0x1.7f7d1cp-20);
    // Cody-and-Waite accurate range reduction. FMA-safe.
    Float y = x;
    y -= z*log2H;
    y -= z*log2L;
    return y;
  }
};



// valid for -87.3365 < x < 88.7228
template<typename Float, int DEGREE, bool FMA>
inline Float __attribute__((always_inline)) unsafe_expf_impl(Float x) {
  using namespace nativeVector;
  using Int = typename toIF<Float>::itype;
  using UInt = typename toIF<Float>::uitype;
  /* Sollya for the following constants:
     display=hexadecimal;
     1b23+1b22;
     single(1/log(2));
     log2H=round(log(2), 16, RN);
     log2L = single(log(2)-log2H);
     log2H; log2L;
     
  */
  // constexpr Float rnd_cst = Float(0xc.p20);
  constexpr float inv_log2f = float(0x1.715476p0);
  
   
  // This is doing round(x*inv_log2f) to the nearest integer
  // Float z = std::floor((x*inv_log2f) +0.5f); Int e = z;
  using std::abs; using nativeVector::abs;
  // exponent 
  Int e = toIF<Float>::convert(abs(x*inv_log2f)-0.5f);
  e = (x>0) ? e : -e;
  Float z = toIF<Float>::convert(e);

  Float y = RangeRedution<Float,FMA>::impl(x,z);


  // we want RN above because it centers the interval around zero
  // but then we could have 2^e = below being infinity when it shouldn't 
  // (when e=128 but p<1)
  // so we avoid this case by reducing e and evaluating a polynomial for 2*exp
  e -=1; 

  // NaN inputs will propagate to the output as expected

  Float p = approx_expf_P<Float,DEGREE>::impl(y);

  // cout << "x=" << x << "  e=" << e << "  y=" << y << "  p=" << p <<"\n";
  UInt biased_exponent= UInt(e+127);
  auto f  =  toIF<Float>::uitof(biased_exponent<<23);
  
  return p * f;
}


#ifndef NO_NATIVEVECTOR

template<typename Float, int DEGREE, bool FMA>
inline Float  __attribute__((always_inline)) unsafe_expf(Float x) {
  return  unsafe_expf_impl<Float,DEGREE, FMA>(x); 
}

template<typename Float, int DEGREE, bool FMA>
inline Float  __attribute__((always_inline)) approx_expf(Float x) {
  using namespace nativeVector;
  constexpr Float zero{0.f};
  constexpr Float inf_threshold = zero+float(0x5.8b90cp4);
  // log of the smallest normal
  constexpr Float zero_threshold_ftz = zero-float(0x5.75628p4); // sollya: single(log(1b-126));
  // flush to zero on the output
  // manage infty output:
  // faster than directly on output!
  // using std::min; using std::max;
  using nativeVector::min; using nativeVector::max;
  x = min(max(x,zero_threshold_ftz),inf_threshold);
  Float r = unsafe_expf<Float,DEGREE,FMA>(x); 

   return r;
}


#else  // only for float...
template<typename Float, int DEGREE, bool FMA>
inline Float unsafe_expf(Float x) {
  return std::exp(x);
}
template<typename Float, int DEGREE, bool FMA>
inline Float approx_expf(Float x) {
  return std::exp(x);
}
#endif  // NO_NATIVEVECTOR




#endif
