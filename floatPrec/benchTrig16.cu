// nvcc -gencode arch=compute_75,code=sm_75 -O3 --expt-relaxed-constexpr -std=c++23 benchTrig16.cu -DNT=512 -DNB=4 -DMX=10000 -DNDEBUG
// ./a.out | grep gtime | cut -d' ' -f6 | tr '\n' ' '

#include "../cuda/clockCuda.h"
#include <cmath>
#include "sincospi.h"
#include "approx_atan2.h"
#include "trig16.h"



struct SCAstdd {
   using Float=double;
   HD_INLINE Float operator()(Float x) {
      auto s = sinpi(x); auto c = cospi(x);
      return atan2(s,c);
   }
};

struct SCAstdf {
   using Float=float;
   HD_INLINE Float operator()(Float x) {
      auto s = sinpif(x); auto c = cospif(x);
      return atan2f(s,c);
   }
};



struct SCAd15 {
   using Float=float;
   HD_INLINE float operator()(float x) {
      auto [s,c] = f32_sincospi(x);
      return unsafe_atan2f<15>(s,c);
   }
};

struct SCAd9 {
   using Float=float;
   HD_INLINE float operator()(float x) {
      auto [s,c] = f32_sincospi(x);
      return unsafe_atan2f<9>(s,c);
   }
};

struct SCAi14 {
  HD_INLINE int16_t operator()(int16_t x) {
    auto [s,c] = trig16::sincos14(x);
    return trig16::atan216(s,c);
  }
};

struct SCAi9 {
  HD_INLINE int16_t operator()(int16_t x) {
    auto [s,c] = trig16::sincos9<true>(x);
    return trig16::atan216(s,c);
  }
};

struct SCAipl {
  HD_INLINE int16_t operator()(int16_t x) {
    auto [s,c] = trig16::sincos9<false>(x);
    return trig16::atan216(s,c);
  }
};


#include<random>
    std::random_device rd;  // a seed source for the random number engine
    std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> rint16(0,65536);
    std::uniform_real_distribution<float> rfloat(-1.f,1.f);

template<typename T>
struct G {
  constexpr T operator()(int i) { return rfloat(gen);}
};


struct GI {
  constexpr int16_t operator()(int i) { return rint16(gen)-32768;}
};


template<typename T>
struct U {
  constexpr T operator()(T x) { return x;}
};



int main() {

  doClock<G<double>,SCAstdd,float>("double std");
  doClock<G<float>,SCAstdf,float>("float std");
  doClock<G<float>,SCAd15,float>("float");
  doClock<G<float>,SCAd9,float>("float 16");
  doClock<GI,SCAi14,int16_t>("int16_t 14");
  doClock<GI,SCAi9,int16_t>("int16_t 9");
  doClock<GI,SCAipl,int16_t>("int16_t pl");

  doClock<G<float>,U<float>,float>("Uf");
  doClock<GI,U<int16_t>,int16_t>("Ui");

  return 0;

}
