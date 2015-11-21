#include<cmath>
#include<limits>
#include<array>
#include<vector>
#include<random>
// c++ -Ofast -fopenmp -mavx2 -mfma NeuNetNaive.cpp  -fopt-info-vec -ftree-loop-if-convert-stores 
// --param max-completely-peel-times=1 

#include <omp.h>
#include <cstdlib>

#include "approx_vexp.h"

constexpr nativeVector::FVect vuno = nativeVector::vzero +1.f;


template<typename T>
T sig(T x) {
  T res =  1.f/(1.f+unsafe_expf<T,3,true>(-x));
  // res = (res > 8.f) ? vuno : res;  
  // res = (res < -8.f) ? vzero : res;
  return res;
  //  return T(1)/(T(1)+std::exp(-x));
}


template<typename T, int N>
struct Neuron {
  std::array<float,N+1> w; 
  T operator()(std::array<T,N> const & x) const {
    using namespace nativeVector;
    input = x;
    T res=vzero +w[N];
    for (int i=0; i<N; ++i) res+=w[i]*x[i];
    return result=sig(res);
  }
  
  T sum(std::array<T,N> const & x) const {
    using namespace nativeVector;
    input = x;
    T res=vzero  +w[N];
    for (int i=0; i<N; ++i) res+=w[i]*x[i];
    return res;
  }
  
  void updateWeight(float learingRate) {
    using namespace nativeVector;
    T corr = learingRate*result*(1.f-result)*error;
    for (int i=0; i<N; ++i) {
      T tmp = corr*input[i];
      for (auto j=0U; j<VSIZE; ++j) w[i] += tmp[j]/float(VSIZE);
    }
    for (auto j=0U; j<VSIZE; ++j) w[N] += corr[j]/float(VSIZE);
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
    using namespace nativeVector;
    output.error = target - (*this)(x);
    for (int i=0; i<M; ++i) middle.neurons[i].error =  output.w[i]*output.error;
    for (int i=0; i<M; ++i) input.neurons[i].error = vzero;
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

#include<ext/random>

template<typename Buffer>
struct Reader {
  explicit Reader(long long iread) : toRead(iread){}
  
  long long operator()(Buffer & buffer, int bufSize) {
    for ( auto & b : buffer) for (int j=0; j<bufSize; j++) b[j]=rgen(eng); // background
    for (auto j=0; j<bufSize; j+=4)
      buffer[4][j] = buffer[3][j] = buffer[2][j] = buffer[0][j];  //signal...

    toRead -= bufSize;
    return toRead;
  }
private:
  long long toRead;
  __gnu_cxx::sfmt19937 eng;
  // std::mt19937 eng;
  std::uniform_real_distribution<float> rgen = std::uniform_real_distribution<float>(0.,1.);

};


#include<iostream>
template<int NX, int MNodes>
void go() {


  using namespace nativeVector;


  long long Nentries = 1024*10000;
  

  // std::mt19937 eng;
  __gnu_cxx::sfmt19937 eng;

  std::uniform_real_distribution<float> rgen(0.,1.);
  std::uniform_real_distribution<float> wgen(-1.,1.);

  // auto vgen = [&]()->FVect { return FVect{wgen(eng),wgen(eng),wgen(eng),wgen(eng), wgen(eng),wgen(eng),wgen(eng),wgen(eng)}; };

  NeuNet<FVect,NX,MNodes> net;

  for (auto & w: net.output.w) w=wgen(eng);
  for (auto & n: net.middle.neurons) for (auto & w: n.w) w=wgen(eng);
  for (auto & n: net.input.neurons) for (auto & w: n.w) w=wgen(eng);
  

  FVect res=vzero;
  double count=0;

  long long tt=0, tc=0;
  constexpr int vsize=VSIZE;
  constexpr unsigned int bufSize=1024;
  // "Struct of Arrays"
  using Data = std::array<float * ,NX>;   // NX columns
  Data buffer; for ( auto & b : buffer) posix_memalign((void**)(&b),sizeof(FVect), sizeof(float)*bufSize); 
//aligned_alloc(sizeof(FVect), sizeof(float)*bufSize);  // yes, leaks...

  // train
  Reader<Data> reader1(Nentries/4);
  while(reader1(buffer,bufSize)>=0) {
    tt -= rdtscp();
    for (auto j=0U; j<bufSize; j+=vsize) {
      std::array<FVect, NX> b; FVect t=vzero; t[0]=1.f; if (vsize>4) t[4]=1.f;
      for(int k=0;k<NX;++k)  b[k] = ((FVect const&)(buffer[k][j]));      
      net.train(b,t,0.02f);
    }
   tt +=rdtscp();
  }


  // classifiy
  Reader<Data> reader2(Nentries);
  while(reader2(buffer,bufSize)>=0) {
    tc -= rdtscp();
    for (auto j=0U; j<bufSize; j+=vsize) {
      std::array<FVect, NX> b;  
      for(int k=0;k<NX;++k)  b[k] = ((FVect const&)(buffer[k][j]));
      res += (net(b)>0.5f) ? vuno : vzero;
      count+=vsize;
    }
    tc +=rdtscp();
  }

  float rr = 0; for (int i=0; i<vsize; ++i) rr+=res[i];
  std::cout << "\nInput Size " << NX << " layer size " << MNodes << std::endl;
  std::cout << "Vector size " << vsize << std::endl;
  std::cout << "total time training " << double(tt)*1.e-9 << std::endl;
  std::cout << "total time classification " << double(tc)*1.e-9 << std::endl;
  std::cout << "final result " << rr/count << std::endl;
}

int main() {

  go<10,14>();
  go<10,7>();

  
  return 0;

}










