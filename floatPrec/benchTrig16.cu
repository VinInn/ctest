// nvcc -gencode arch=compute_75,code=sm_75 -O3 --expt-relaxed-constexpr -std=c++23 benchTrig16.cu -DNT=512 -DNB=4 -DMX=10000
// ./a.out | grep gtime | cut -d' ' -f6 | tr '\n' ' '

#include "../cuda/clockCuda.h"
#include <cmath>
#include "sincospi.h"
#include "approx_atan2.h"
#include "trig16.h"



struct SCAstd {
   using Float=float;
   HD_INLINE float operator()(float x) {
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



template<typename T>
struct G {
  constexpr T operator()(int i) { return T(2)*T(i)/T(NB*NT) -T(1);}
};

template<typename T>
struct U {
  constexpr T operator()(T x) { return x;}
};



int main() {

  doClock<G<float>,SCAstd,float>("float std");
  doClock<G<float>,SCAd15,float>("float");
  doClock<G<float>,SCAd9,float>("float 16");

  doClock<G<float>,U<float>,float>("Uf");

  return 0;

}
