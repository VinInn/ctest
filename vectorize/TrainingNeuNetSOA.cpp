#include<cmath>
#include<limits>
#include<array>
#include<vector>
#include<random>
// c++ -Ofast -fopenmp -mavx2 -mfma NeuNetNaive.cpp  -fopt-info-vec -ftree-loop-if-convert-stores 
// --param max-completely-peel-times=1 

#include <omp.h>

#include "../floatPrec/approx_vexp.h"

constexpr approx_math::float32x8_t zero8{0,0,0,0,0,0,0,0};
constexpr approx_math::float32x8_t uno8{1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f};
constexpr int VSIZE = 8;

template<typename T>
T sig(T x) {
  T res =  1.f/(1.f+unsafe_expf<T,3,true>(-x));
  //  res = (res > 8.f) ? uno8 : res;  
  //  res = (res < -8.f) ? zero8 : res;
  return res;
  //  return T(1)/(T(1)+std::exp(-x));
}


template<typename T, int N>
struct Neuron {
  std::array<float,N+1> w; 
  T operator()(std::array<T,N> const & x) const {
    input = x;
    T res=zero8 +w[N];
    for (int i=0; i<N; ++i) res+=w[i]*x[i];
    return result=sig(res);
  }
  
  T sum(std::array<T,N> const & x) const {
    input = x;
    T res=zero8  +w[N];
    for (int i=0; i<N; ++i) res+=w[i]*x[i];
    return res;
  }
  
  void updateWeight(float learingRate) {
    T corr = learingRate*result*(1.f-result)*error;
    for (int i=0; i<N; ++i) {
      T tmp = corr*input[i];
      for (int j=0; j<VSIZE; ++j) w[i] += tmp[j]/float(VSIZE);
    }
    for (int j=0; j<VSIZE; ++j) w[N] += corr[j]/float(VSIZE);
  }
  mutable std::array<T,N> input;
  mutable T result;
  T error;
};


template<typename T, int N, int M>
struct Layer {
  std::array<Neuron<T,N>,M> neurons;
  using Output = std::array<T,M>;
  Output operator()(std::array<T,N> const & x) const {
    Output res;
    for (int i=0; i<M; ++i) res[i] = neurons[i](x);
    // for (int i=0; i<M; ++i) res[i] = neurons[i].sum(x);
    // for (int i=0; i<M; ++i) res[i] = sig(res[i]);

    return res;
  }

  void updateWeight(float learingRate) {
    for (auto & n : neurons) n.updateWeight(learingRate);
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

  void train(std::array<T,N> const & x, T const & target, float learingRate) {
    output.error = target - (*this)(x);
    for (int i=0; i<M; ++i) middle.neurons[i].error =  output.w[i]*output.error;
    for (int i=0; i<M; ++i) input.neurons[i].error = zero8;
    for (int j=0; j<M; ++j) for (int i=0; i<M; ++i) input.neurons[j].error += middle.neurons[i].w[j]*middle.neurons[i].error;
    input.updateWeight(learingRate);
    middle.updateWeight(learingRate);
    output.updateWeight(learingRate);
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

  using namespace approx_math;

  long long Nentries = 1024*10000;
  

  std::mt19937 eng;
  std::uniform_real_distribution<float> rgen(0.,1.);
  std::uniform_real_distribution<float> wgen(-1.,1.);

  // auto vgen = [&]()->float32x8_t { return float32x8_t{wgen(eng),wgen(eng),wgen(eng),wgen(eng), wgen(eng),wgen(eng),wgen(eng),wgen(eng)}; };

  NeuNet<float32x8_t,NX,MNodes> net;

  for (auto & w: net.output.w) w=wgen(eng);
  for (auto & n: net.middle.neurons) for (auto & w: n.w) w=wgen(eng);
  for (auto & n: net.input.neurons) for (auto & w: n.w) w=wgen(eng);
  

  float32x8_t res=zero8;
  double count=0;

  long long t=0;
  constexpr int vsize=8;
  constexpr unsigned int bufSize=1024;
  using Data = std::array<std::vector<float>,NX>; 
  Data buffer; for ( auto & b : buffer) b.resize(bufSize);

  // train
  for (int iter=0; iter<1; ++iter)
  for (int kk=0; kk<Nentries/4; kk+=bufSize) {
    int ll=0;
    for ( auto & b : buffer) for (auto & e : b) e=rgen(eng); // background
    for (int j=0; j<bufSize; j+=4)
      buffer[4][j] = buffer[3][j] = buffer[2][j] = buffer[0][j];  //signal...
    for (int j=0; j<bufSize; j+=vsize) {
      std::array<float32x8_t, NX> b; float32x8_t t=zero8; t[0]=1.f; t[4]=1.f;
      for(int k=0;k<NX;++k) for (int i=0; i<vsize; ++i) b[k][i] = buffer[k][j+i];      
      net.train(b,t,0.02f);
    }
  }
  
  for (int kk=0; kk<Nentries; kk+=bufSize) {
    int ll=0;
    for ( auto & b : buffer) for (auto & e : b) e=rgen(eng); // background
    for (int j=0; j<bufSize; j+=4)
      buffer[4][j] = buffer[3][j] = buffer[2][j] = buffer[0][j];  //signal... 
    t -= rdtscp();
    for (int j=0; j<bufSize; j+=vsize) {
      std::array<float32x8_t, NX> b;  
      for(int k=0;k<NX;++k) for (int i=0; i<vsize; ++i) b[k][i] = buffer[k][j+i];
      res += (net(b)>0.5f) ? uno8 : zero8;
      count+=vsize;
    }
    t +=rdtscp();

  }

  float rr = 0; for (int i=0; i<vsize; ++i) rr+=res[i];
  std::cout << "\nInput Size " << NX << " layer size " << MNodes << std::endl;
  std::cout << "total time " << double(t)*1.e-9 << std::endl;
  std::cout << "final result " << rr/count << std::endl;
}

int main() {

  go<10,14>();
  go<10,7>();

  
  return 0;

}










