// nvcc -gencode arch=compute_75,code=sm_75 -O3 --expt-relaxed-constexpr -std=c++23 benchCoth.cu -DNT=512 -DNB=4 -DMX=10000
// ./a.out | grep gtime | cut -d' ' -f6 | tr '\n' ' '

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
     x = std::abs(x);
     return 
     T(0.999999986449) + x*(T(2.06961523705e-6) + x*( T(-0.500073072319) + x*( T(0.00106750423055) + x*( T(0.200707822535) + x*( T(0.0295750777896) + x*( T(-0.150307883454) + x*( T(0.0821370891714) + x*T(-0.0150543655971)
      )))))));
   }
};


template<typename T>
struct pade {
   using Float=T;
   HD_INLINE T operator()(T x) {
     x = std::abs(x)- T(0.6);
     T p = T(0.843551) + x*(T(-0.0488014) + x*(T(-0.0333239) + x*(T(0.00191492) + x*T(0.000527805) )));
     T q = T(1.0) + x*(T(0.479197) + x*(T(0.429426) + x*(T(0.0416364) + x*T(0.0185811)  )));
     return p/q;
   }
};

#include "Horner.h"


namespace sech9_c {
  HOST_DEVICE_CONSTANT float c[10] = {0.00484524607845, -0.0414213996033, 0.141985913707, -0.223653720345, 0.0820334481706, 0.178636110405, 
                                      0.00629169010252, -0.500693216667, 3.04210804006e-5, 0.999999773462};
}
struct sech9 {
   using Float=float;
   HD_INLINE float operator()(float x) {
     return horner<9>(x,sech9_c::c);
   }
};


namespace sech9_d {
  HOST_DEVICE_CONSTANT double c[10] = {0.00484524607845, -0.0414213996033, 0.141985913707, -0.223653720345, 0.0820334481706, 0.178636110405,
                                      0.00629169010252, -0.500693216667, 3.04210804006e-5, 0.999999773462};
}
struct sech9d {
   using Float=double;
   HD_INLINE Float operator()(Float x) {
     return horner<9>(x,sech9_d::c);
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
  constexpr T operator()(int i) { return T(i)*T(1.e-4);}
};

template<typename T>
struct U {
  constexpr T operator()(T x) { return x;}
};


#include "Exp16.h"

template<typename EXP>
struct sechI {
  HD_INLINE sechI(){}
  HD_INLINE void init(double r=5.) { exp16 = EXP(r); }
  EXP exp16;
  HD_INLINE float operator()(int x){ return 2.f/(exp16.pexp(x)+exp16.nexp(x)); }
};

template<>
HD_INLINE void init<sechI<Exp16_2>>(sechI<Exp16_2> & f) { f.init();}
template<>
HD_INLINE void init<sechI<Exp16_4>>(sechI<Exp16_4> & f) { f.init();}



#include "LUT16.h"

__device__ LUT16 lutP;

__device__ LUT16 lut1P;
__device__ LUT16 lut2P;
__device__ LUT16 lut3P;
__device__ LUT16 lut4P;

double pexp(double x) { return exp(x);}
double nexp(double x) { return exp(-x);}
double logs2(double x) { return log(2*x+1);}

struct sechL {
  HD_INLINE sechL(){}
  HD_INLINE void init() { 
  }
  HD_INLINE float operator()(int x){ return lutP[x]; }
};

struct sech4L {
  HD_INLINE sech4L(){}
  HD_INLINE void init() {
  }
  HD_INLINE float operator()(int x){ return lut1P[x]+lut2P[x]+lut3P[x]+lut4P[x]; }
};

struct GI {
  GI()  {
    LUT16 lut(secosh<double>(),5.);
    cudaMemcpyToSymbol(lutP,&lut,sizeof(LUT16));
    LUT16 lut1(secosh<double>(),5.);
    cudaMemcpyToSymbol(lut1P,&lut1,sizeof(LUT16));
    LUT16 lut2(nexp,5.);
    cudaMemcpyToSymbol(lut2P,&lut2,sizeof(LUT16));
    LUT16 lut3(pexp,5.);
    cudaMemcpyToSymbol(lut3P,&lut3,sizeof(LUT16));
    LUT16 lut4(logs2,5.);
    cudaMemcpyToSymbol(lut4P,&lut4,sizeof(LUT16));
  }
  HD_INLINE float operator()(int i) { return 14*i;}
};


int main() {

  doClock<G<double>,naive<double>,double>("naived");
  doClock<G<float>,naive<float>,float>("naivef");
  doClock<G<double>,secosh<double>,double>("secoshd");
  doClock<G<float>,secosh<float>,float>("secoshf");

  doClock<G<double>,poly<double>,double>("polyd");
  doClock<G<float>,poly<float>,float>("polyf");

  doClock<G<double>,sech9d,double>("sech9d");
  doClock<G<float>,sech9,float>("sech9");
  doClock<G<float>,sech5,float>("sech5");

  doClock<GI,sechI<Exp16_2>,float,int>("int LUT2");
  doClock<GI,sechI<Exp16_4>,float,int>("int LUT4");
  doClock<GI,sechL,float,int>("int LUT");
  doClock<GI,sech4L,float,int>("int 4LUT");

  doClock<G<double>,U<double>,double>("Ud");
  doClock<G<float>,U<float>,float>("Uf");

   return 0;
}
