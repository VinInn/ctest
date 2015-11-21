#ifndef APPROX_LOG_H
#define APPROX_LOG_H
/*  Quick and dirty, branchless, log implementations
    Author: Florent de Dinechin, Aric, ENS-Lyon 
    All right reserved

Warning + disclaimers:
- no special case handling (infinite/NaN inputs, even zero input, etc)
- no input subnormal handling, you'll get completely wrong results.
  This is the worst problem IMHO (leading to very rare but very bad bugs)
  However it is probable you can guarantee that your input numbers 
  are never subnormal, check that. Otherwise I'll fix it...
- output accuracy reported is only absolute. 
  Relative accuracy may be arbitrary bad around log(1), 
  especially for approx_log0. approx_logf is more or less OK.
- The larger/smaller the input x (i.e. away from 1), the better the accuracy.
- For the higher degree polynomials it is possible to win a few cycles 
  by parallelizing the evaluation of the polynomial (Estrin). 
  It doesn't make much sense if you want to make a vector function. 
- All this code is FMA-safe (and accelerated by FMA)
 
Feel free to distribute or insert in other programs etc, as long as this notice is attached.
    Comments, requests etc: Florent.de.Dinechin@ens-lyon.fr

Polynomials were obtained using Sollya scripts (in comments): 
please also keep these comments attached to the code of approx_logf. 
*/


#include "nativeVector.h"


template<typename Float, int DEGREE>
struct approx_logf_P {  static Float impl(Float p); };


// the following is Sollya output

// degree =  2   => absolute accuracy is  7 bits
template<typename Float>
struct approx_logf_P<Float,2> {
  static Float impl(Float y) {
    return  y * ( float(0x1.0671c4p0) + y * ( float(-0x7.27744p-4) )) ;
  }
};

// degree =  3   => absolute accuracy is  10 bits
template<typename Float>
struct approx_logf_P<Float,3> {
  static Float impl(Float y) {
    return  y * (float(0x1.013354p0) + y * (-float(0x8.33006p-4) + y * float(0x4.0d16cp-4))) ;
  }				  
};
// degree =  4   => absolute accuracy is  13 bits
template<typename Float>
struct approx_logf_P<Float,4> {
  static Float impl(Float y) {
    return  y * (float(0xf.ff5bap-4) + y * (-float(0x8.13e5ep-4) + y * (float(0x5.826ep-4) + y * (-float(0x2.e87fb8p-4))))) ;
  }
};
// degree =  5   => absolute accuracy is  16 bits
template<typename Float>
struct approx_logf_P<Float,5> {
  static Float impl(Float y) {
  return  y * (float(0xf.ff652p-4) + y * (-float(0x8.0048ap-4) + y * (float(0x5.72782p-4) + y * (-float(0x4.20904p-4) + y * float(0x2.1d7fd8p-4))))) ;
}
};
// degree =  6   => absolute accuracy is  19 bits
template<typename Float>
struct approx_logf_P<Float,6> {
  static Float impl(Float y) {
  return  y * (float(0xf.fff14p-4) + y * (-float(0x7.ff4bfp-4) + y * (float(0x5.582f6p-4) + y * (-float(0x4.1dcf2p-4) + y * (float(0x3.3863f8p-4) + y * (-float(0x1.9288d4p-4))))))) ;
}
};
// degree =  7   => absolute accuracy is  21 bits
template<typename Float>
struct approx_logf_P<Float,7> {
  static Float impl(Float y) {
  return  y * (float(0x1.000034p0) + y * (-float(0x7.ffe57p-4) + y * (float(0x5.5422ep-4) + y * (-float(0x4.037a6p-4) + y * (float(0x3.541c88p-4) + y * (-float(0x2.af842p-4) + y * float(0x1.48b3d8p-4))))))) ;
}
};
// degree =  8   => absolute accuracy is  24 bits
template<typename Float>
struct approx_logf_P<Float,8> {
  static Float impl(Float y) {
   return  y * ( float(0x1.00000cp0) + y * (float(-0x8.0003p-4) + y * (float(0x5.55087p-4) + y * ( float(-0x3.fedcep-4) + y * (float(0x3.3a1dap-4) + y * (float(-0x2.cb55fp-4) + y * (float(0x2.38831p-4) + y * (float(-0xf.e87cap-8) )))))))) ;
}
};

template<typename Float, int DEGREE>
inline Float __attribute__((always_inline)) unsafe_logf_impl(Float x) {
  using namespace nativeVector;
  using Int = typename toIF<Float>::itype;
  using UInt = typename toIF<Float>::uitype;

  Int ix = toIF<Float>::ftoi(x);
  
  // as many integer computations as possible, most are 1-cycle only, and lots of ILP.
  Int e= (((ix) >> 23) & 0xFF) -127; // extract exponent
  Int m = (ix & 0x007FFFFF) | 0x3F800000; // extract mantissa as an FP number
  
  Int adjust = (ix>>22)&1; // first bit of the mantissa, tells us if 1.m > 1.5
  m -= (adjust << 23); // if so, divide 1.m by 2 (exact operation, no rounding)
  e += adjust;           // and update exponent so we still have x=2^E*y
  
  // now back to floating-point
  Float y = toIF<Float>::itof(m) -1.0f; // Sterbenz-exact; cancels but we don't care about output relative error
  // all the computations so far were free of rounding errors...

  // the following is based on Sollya output
  Float p = approx_logf_P<Float,DEGREE>::impl(y);
  

  constexpr float Log2=0xb.17218p-4; // 0.693147182464599609375 
  return toIF<Float>::convert(e)*Log2+p;
}


template<typename Float, int DEGREE>
inline Float  __attribute__((always_inline)) unsafe_logf(Float x) {
  return  unsafe_logf_impl<Float,DEGREE>(x); 
}

template<typename Float, int DEGREE>
inline Float  __attribute__((always_inline)) approx_logf(Float x) {
  return unsafe_logf<Float,DEGREE>(x);
}

#endif
