#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>

#include<iostream>


inline
float tanhP(float y) {
  y = std::min(y,8.f);
 if (y<1)
   return  -0x1.p-24 + y * (0x1.0000ccp0 + y * (-0x1.5bbb04p-12 + y * (-0x5.4783dp-4 + y * (-0x4.4bab1p-8 + y * (0x2.dabdfp-4 + y * (-0x1.0a3a3p-4 + y * (-0x3.7d3528p-8 + y * 0x2.45a8f8p-8))))))) ;
 else if(y<2)
   return 0x4.p-8 + y * (0xe.168a8p-4 + y * (0x6.31ed2p-4 + y * (-0x1.057f18p0 + y * (0xb.5bd26p-4 + y * (-0x4.137b8p-4 + y * (0xc.699dcp-8 + y * (-0xf.f984cp-12))))))) ;
 else if(y<4)
   return -0x4.p-4 + y * (0x1.e83e2cp0 + y * (-0x1.4b1074p0 + y * (0x8.02dacp-4 + y * (-0x1.e579e8p-4 + y * (0x4.5b108p-8 + y * (-0x5.92931p-12 + y * 0x3.099c98p-16)))))) ;
 else
   return 0x8.p-4 + y * (0x9.1faccp-4 + y * (-0x4.795dp-4 + y * (0x1.383f84p-4 + y * (-0x3.302788p-8 + y * (0x4.fc751p-12 + y * (-0x4.50e1ap-16 + y * 0x1.9801ep-20)))))) ;
}

inline
float tanhP4(float y) {
  y = std::min(y,8.f);
  // return y* (float(1.+0x4.p-8) + y * (float(-0x2.984ecp-4) + y * (float(-0x2.4389ep-4) + y * (float(0xf.e4316p-8) + y * (float(-0x2.47718p-8) + y * float(0x1.c5377cp-12)))))) ;
  float ret=1;
 if (y<2.f)
   ret = (y<1.f) ?  // float(-0x2.p-16) + y * (float(0x1.001214p0) + y * (float(0x1.cf21ccp-8) + y * (float(-0x6.4149bp-4) + y * float(0x2.5302ep-4)))) :
                  y* (float(1.-0x8.p-16) + y * (float(0x1.750e28p-8) + y * (float(-0x6.0f27cp-4) + y * (float(0x1.fd55f8p-4) + y * float(0x2.aed238p-8))))) :
                  float(-0x2.p-4) + y * (float(0x1.840d28p0) + y * (float(-0xc.f8a0ap-4) + y * (float(0x3.34fc6p-4) + y * (float(-0x4.da39fp-8))))) ;
 else
   ret = (y<4.f) ? float(0x1.p-4) + y * (float(0x1.36603p0) + y * (float(-0xa.5e5fep-4) + y * (float(0x2.d74cfp-4) + y * (float(-0x6.57f4p-8) + y * float(0x5.bdd12p-12))))) :
                 float(0x1.p0) + y * (float(-0xb.596f8p-12) + y * (float(0x5.35b3fp-12) + y * (float(-0xc.9dee8p-16) + y * float(0xa.1456p-20)))) ;
 return ret;
}

inline
float dirty_tanh(float x) {
   return std::copysign(tanhP(std::abs(x)),x);
}

inline
float approx_tanh(float x) {
   return std::copysign(tanhP4(std::abs(x)),x);
}


#include<cmath>
#include<iostream>


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

// performance test
#include <x86intrin.h>
inline volatile unsigned long long rdtsc() {
 unsigned int i = 0;
 return __rdtscp(&i);
}


namespace justcomp {
  constexpr int NN=1024*1024;
  float a[NN], b[NN];
  void ap() {
    for (int i=0; i!=NN; ++i)
      b[i] = approx_tanh(a[i]);
  }
  void dy() {
    for (int i=0; i!=NN; ++i)
      b[i] = dirty_tanh(a[i]);
  }
  void st() {
    for (int i=0; i!=NN; ++i)
      b[i] = std::tanh(a[i]);
  }

}

void perf() {
  using namespace approx_math;
  unsigned long long ta=0;
  unsigned long long td=0;
  unsigned long long ts=0;

  binary32 x,r;
  float sum=0;
  long long ntot=0;
  x.f=1.e-20; // should be 0 but 
  while (x.f<8) { // this is 5*2^23 tests
    ++ntot;
    int i=0;
    while(i<justcomp::NN) { 
      x.ui32++;
      justcomp::a[i++]=x.f;
    }
   ta -= rdtsc();
   justcomp::ap();
   ta += rdtsc();
   td -= rdtsc();
   justcomp::dy();
   td += rdtsc();
   ts -= rdtsc();
   justcomp::st();
   ts += rdtsc();

   for (int i=0; i!=justcomp::NN; ++i)
      sum += justcomp::b[i];
  }
  std::cout << "times " << double(ta)/double(justcomp::NN*ntot) << std::endl;
  std::cout << "times " << double(td)/double(justcomp::NN*ntot) << std::endl;
  std::cout << "times " << double(ts)/double(justcomp::NN*ntot) << std::endl;
  std::cout << "sum= " << sum << " to prevent compiler optim." << std::endl;
}


int main() {

  for (float x = -10; x<10; x+=0.5)
    std::cout << x << " " << std::tanh(x) << " " << dirty_tanh(x) << " " << approx_tanh(x) 
	      << " " << std::tanh(x)-dirty_tanh(x)  << " " << std::tanh(x)-approx_tanh(x) << std::endl;

 for (float x = 0.0001; x<1; x*=10)
    std::cout << x << " " << std::tanh(x) << " " << dirty_tanh(x) << " " << approx_tanh(x)
              << " " << std::tanh(x)-dirty_tanh(x)  << " " << std::tanh(x)-approx_tanh(x) << std::endl;
 for (float x = 0.0; x<1.1; x+=.05)
    std::cout << x << " " << std::tanh(x) << " " << dirty_tanh(x) << " " << approx_tanh(x)
              << " " << std::tanh(x)-dirty_tanh(x)  << " " << std::tanh(x)-approx_tanh(x) << std::endl;


perf();

  return 0;

}
