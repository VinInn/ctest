#include<cmath>
#include<limits>
#include<array>
#include<vector>
#include<random>
// c++ -Ofast -fopenmp -mavx2 -mfma NeuNetNaive.cpp  -fopt-info-vec -ftree-loop-if-convert-stores 
// --param max-completely-peel-times=1 

#include <omp.h>

#include "../floatPrec/approx_exp.h"


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
float fast_tanh(float x) {
   return std::copysign(tanhP4(std::abs(x)),x);
}


template<typename T>
T sig(T x) {
  return T(1)/(T(1)+unsafe_expf<3>(x));
  //  return T(1)/(T(1)+std::exp(-x));
  // return fast_tanh(x);
}


template<typename T, int N>
struct Neuron {
  std::array<T,N> w; 
  T operator()(std::array<T,N> const & x) const {
    T res=0;
    for (int i=0; i<N; ++i) res+=w[i]*x[i];
    return sig(res);
  }
  
  T sum(std::array<T,N> const & x) const {
    T res=0;
    for (int i=0; i<N; ++i) res+=w[i]*x[i];
    return res;
  }
};
template<typename T, int N, int M>
struct Layer {
  std::array<Neuron<T,N>,M> neurons;
  using Output = std::array<T,M>;
  Output operator()(std::array<T,N> const & x) const {
    Output res;
    //    for (int i=0; i<M; ++i) res[i] = neurons[i](x);
    for (int i=0; i<M; ++i) res[i] = neurons[i].sum(x);
    for (int i=0; i<M; ++i) res[i] = sig(res[i]);

    return res;
  }

};




template<typename T, int N, int M>
struct NeuNet {
  Layer<T,N,M> input;
  Layer<T,M,M> middle;
  Neuron<T,M> output;
  T operator()(std::array<T,N> const & x) {
    return output(middle(input(x)));
  }

};

#include <x86intrin.h>
unsigned int taux=0;
inline unsigned long long rdtscp() {
 return __rdtscp(&taux);
}


#include<iostream>
template<int NX, int MNodes>
void go() {


  long long Nentries = 1024*1000;
  

  std::mt19937 eng;
  std::uniform_real_distribution<float> rgen(0.,1.);
  std::uniform_real_distribution<float> wgen(-1.,1.);


  NeuNet<float,NX,MNodes> net;

  for (auto & w: net.output.w) w=wgen(eng);
  for (auto & n: net.middle.neurons) for (auto & w: n.w) w=wgen(eng);
  for (auto & n: net.input.neurons) for (auto & w: n.w) w=wgen(eng);
  

  using Data = std::array<float,NX>; 
  

  float res=0;
  double count=0;

  long long t=0;
  constexpr unsigned int bufSize=1024;
  std::vector<Data> buffer(bufSize);
  for (int kk=0; kk<Nentries; kk+=bufSize) {
    for ( auto & b : buffer) for (auto & e : b) e=rgen(eng);
    t -= rdtscp();
    for ( auto & b : buffer) {
      res += net(b);
      ++count;
    }
    t +=rdtscp();

  }
    
  std::cout << "\nInput Size " << NX << " layer size " << MNodes << std::endl;
  std::cout << "total time " << double(t)*1.e-9 << std::endl;
  std::cout << "final result " << res/count << std::endl;
}

int main() {

  go<10,14>();
  go<10,7>();

  
  return 0;

}










