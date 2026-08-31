// nvcc -gencode arch=compute_75,code=sm_75 -O3 --expt-relaxed-constexpr -std=c++23 benchCosh.cu -DNT=512 -DNB=4 -DMX=10000


#include "../cuda/clockCuda.h"
#include <cmath>
#include "Horner.h"



template<typename T>
struct naive {
   using Float=T;
   HD_INLINE T operator()(T x) { return std::sin(T(2)*std::atan(std::exp(x)));}
};

template<typename T>
struct secosh {
   using Float=T;
   HD_INLINE T operator()(T x) { return T(1)/std::cosh(x);}
};


template<typename T>
struct poly {
   using Float=T;
   HD_INLINE T operator()(T x) {
     x = std::abs(x)- T(0.6);
     return 
     T(0.999999986449) + x*(T(2.06961523705e-6) + x*( T(-0.500073072319) + x*( T(0.00106750423055) + x*( T(0.200707822535) + x*( T(0.0295750777896) + x*( T(-0.150307883454) + x*( T(0.0821370891714) + x*T(-0.0150543655971)
      )))))));
   }
};


template<typename T>
struct pade {
   using Float=T;
   HD_INLINE T operator()(T x) {
     x = std::abs(x);
     T p = T(0.843551) + x*(T(-0.0488014) + x*(T(-0.0333239) + x*(T(0.00191492) + x*T(0.000527805) )));
     T q = T(1.0) + x*(T(0.479197) + x*(T(0.429426) + x*(T(0.0416364) + x*T(0.0185811)  )));
     return p/q;
   }
};

#include "Horner.h"


namespace sech9_c {
  HOST_DEVICE_CONSTANT float c[10] = {0.00484524607845, -0.0414213996033, 0.141985913707, -0.223653720345, 0.0820334481706, 0.178636110405, 0.00629169010252, -0.500693216667, 3.04210804006e-5, 0.999999773462};
}
struct sech9 {
   using Float=float;
   HD_INLINE float operator()(float x) {
     return horner<9>(x,sech9_c::c);
   }
};

/*
struct sech8 {
   using Float=float;
HD_INLINE float operator()(float x) {
  HOST_DEVICE_CONSTANT float c[9] = {-0.00879617718167, 0.0505179227683, -0.0855020983306, -0.0394092119823, 0.24122061376, -0.0117719227185, -0.498093455092, -0.000112573426959, 1.00000105203};
  return horner<8>(x,c);
}
};

struct sech6 {
   using Float=float;
HD_INLINE float operator() (float x) {
  HOST_DEVICE_CONSTANT float c[7] = {0.0325071921628, -0.183832500115, 0.340654638413, -0.0501527552854, -0.490379984351, -0.000761535402113, 1.00001123574};
  return horner<6>(x,c);
}
};
*/


namespace sech5_c {
  HOST_DEVICE_CONSTANT float c[6] = {-0.0371366267403, 0.0924264377937, 0.143342769817, -0.558449420714, 0.00795787347873, 0.999832461682};
}
struct sech5 {
  using Float=float;
  HD_INLINE float operator() (float x) {
    return horner<5>(x,sech5_c::c);
  }
};


template<typename T>
struct G {
  constexpr T operator()(int i) { return T(i)*T(1.e-7);}
};

template<typename T>
struct U {
  constexpr T operator()(T x) { return x;}
};

/*
#include "Exp16.h"
Exp16 exp16(5.);

void loop16() {
     for (int i=0; i<4*1024;i++) {
       for(int j=0; j<1024; ++j)
         fout[j] =  2.f/(exp16.pexp(iin[j])+exp16.nexp(iin[j]));
     }
   }
}
*/

int main() {

  doClock<G<double>,naive<double>,double>("naived");
  doClock<G<float>,naive<float>,float>("naivef");
  doClock<G<double>,secosh<double>,double>("secoshd");
  doClock<G<float>,secosh<float>,float>("secoshf");
  doClock<G<double>,poly<double>,double>("polyd");
  doClock<G<float>,poly<float>,float>("polyf");
  doClock<G<float>,sech9,float>("sech9");
  doClock<G<float>,sech5,float>("sech5");
  doClock<G<double>,U<double>,double>("Ud");
  doClock<G<float>,U<float>,float>("Uf");

   return 0;
}
