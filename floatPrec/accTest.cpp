#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>

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
}
#endif


#include<typeinfo>
#include<iostream>
template<typename STD, typename APPROX>
void accTest(STD stdf, APPROX approx, float mm=std::numeric_limits<float>::min(), float mx=std::numeric_limits<float>::max()) {
  using namespace approx_math;
  std::cout << std::endl << "launching  exhaustive test for " << typeid(APPROX).name() << std::endl;
  binary32 x,pend, r,ref;
  int maxdiff=0;
  int n127=0;
  int n16393=0;
  float ad=0., rd=0;
  x.f=mm;
  x.ui32++;
  pend.f=mx;
  pend.ui32--;
  std::cout << "limits " << x.f << " " << pend.f << " " << pend.ui32-x.ui32 << std::endl;
  while(x.ui32<pend.ui32) {
    x.ui32++;
    r.f=approx(x.f);
    ref.f=stdf(x.f); // double-prec one  (no hope with -fno-math-errno)
    ad = std::max(ad,std::abs(r.f-ref.f));
    rd = std::max(rd,std::abs((r.f/ref.f)-1.f));
    int d=abs(r.i32-ref.i32);
    if(d>maxdiff) {
      // std::cout << "new maxdiff for x=" << x.f << " : " << d << std::endl;
      maxdiff=d;
	}
    if (d>127) ++n127;
    if (d>16393) ++n16393;
  }
  std::cout << "absdiff / reldeff/ maxdiff / diff >127 / diff >16393 :  " << ad << " / " << rd << " / "  
	    << maxdiff << " / " << n127<< " / " << n16393<< std::endl;
}


#include "approx_exp.h"
#include "approx_log.h"

// inline
float tanhP4(float y) {
  y = std::min(y,8.f);
  // return y* (float(1.+0x4.p-8) + y * (float(-0x2.984ecp-4) + y * (float(-0x2.4389ep-4) + y * (float(0xf.e4316p-8) + y * (float(-0x2.47718p-8) + y * float(0x1.c5377cp-12)))))) ;
  float ret=1;
 if (y<2.f)
   ret = (y<1.f) ?  // float(-0x2.p-16) + y * (float(0x1.001214p0) + y * (float(0x1.cf21ccp-8) + y * (float(-0x6.4149bp-4) + y * float(0x2.5302ep-4)))) :
                  y* (float(0x1.fffp-1) + y * (float(0x1.750e28p-8) + y * (float(-0x6.0f27cp-4) + y * (float(0x1.fd55f8p-4) + y * float(0x2.aed238p-8))))) :
                  float(-0x2.p-4) + y * (float(0x1.840d28p0) + y * (float(-0xc.f8a0ap-4) + y * (float(0x3.34fc6p-4) - y * (float(0x4.da39fp-8))))) ;
 else
   ret = (y<4.f) ? float(0x1.p-4) + y * (float(0x1.36603p0) + y * (float(-0xa.5e5fep-4) + y * (float(0x2.d74cfp-4) - y * (float(0x6.57f4p-8) - y * float(0x5.bdd12p-12))))) :
                 float(0x1.p0) + y * (float(-0xb.596f8p-12) + y * (float(0x5.35b3fp-12) - y * (float(0xc.9dee8p-16) - y * float(0xa.1456p-20)))) ;
 return ret;
}


float logE(float x) {
  return std::log(1.f-std::exp(-x));
}

float logE4(float y) {
  const float log2 = std::log(2);
  return (y<log2) ? -0x2.p-8 + y * (-0x3.917bdp0 + y * (0x1.03954cp4 + y * (-0x2.c2de08p4 + y * (0x3.c0ed6p4 + y * (-0x1.ef10bp4))))) 
    :  y*(-0x8.p0 + y * (0x1.6a07c8p4 + y * (-0x1.c0a04p4 + y * (0x1.2218dp4 + y * (-0x5.f6ecfp0 + y * 0xc.9bbep-4))))) ;
    // -0x2.p0 + y * (0x3.0a88b8p0 + y * (-0x2.2ea5f8p0 + y * (0xd.6112cp-4 + y * (-0x2.96e18p-4 + y * 0x3.29a5ep-8)))) ;

}


float erf4(float x) {
  auto xx = std::min(std::abs(x),5.f);
  xx*=xx;
  return std::copysign(std::sqrt(1.f- unsafe_expf<4>(-xx*(1.f+0.2733f/(1.f+0.147f*xx)) )),x);
  // return std::sqrt(1.f- std::exp(-x*x*(1.f+0.2733f/(1.f+0.147f*x*x)) ));
}

#include<cstdio>
int main() {

  printf("%a\n",1.-0x8.p-16);

  //  accTest(logE,logE4, std::numeric_limits<float>::min()*10000, 4 );
  accTest(::erf,erf4, .01, 8  );


  accTest(logE,logE4, std::log(2), 2 );

  accTest(::tanh,tanhP4, std::numeric_limits<float>::min()*10000, 8 );

  accTest(::exp,approx_expf<3>, std::numeric_limits<float>::min(), 80 );
  accTest(::log,approx_logf<4>,std::numeric_limits<float>::min(),std::numeric_limits<float>::max() );

  return 0;

}
