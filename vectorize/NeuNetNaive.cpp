#include<cmath>
#include<limits>
#include<array>
#include<vector>
#include<random>
// c++ -Ofast -fopenmp -mavx2 -mfma NeuNetNaive.cpp  -fopt-info-vec -ftree-loop-if-convert-stores 
// --param max-completely-peel-times=1 

#include <omp.h>


// #define SCALAR
#include "../floatPrec/approx_vexp.h"


template<typename T>
T sigmoid(T x) {
  return T(1)/(T(1)+unsafe_expf<T,5,true>(-x));
  //  return T(1)/(T(1)+std::exp(-x));
}


template<typename T, int N>
struct Neuron {
  std::array<float,N+1> w; 
  T operator()(std::array<T,N> const & x) const {
    input = x;
    T res = w[N];
    for (int i=0; i<N; ++i) res+=w[i]*x[i];
    return result=sigmoid(res);
  }
  
  T sum(std::array<T,N> const & x) const {
    input = x;
    T res = w[N];
    for (int i=0; i<N; ++i) res+=w[i]*x[i];
    return res;
  }
  
  void updateWeight(float learingRate) {
    T corr = learingRate*result*(1.f-result)*error;
    for (int i=0; i<N; ++i)  w[i] += corr*input[i];
    w[N] += corr;
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
    return res;
  }

  void updateWeight(float learingRate) {
    for (auto & n : neurons) n.updateWeight(learingRate);
  }
  
};

template<typename T, int N, int M>
struct NeuNet {
  Layer<T,N,M> hidden1;
  Layer<T,M,M> hidden2;
  Neuron<T,M> output;

  T operator()(std::array<T,N> const & x) const {
    return output(hidden2(hidden1(x)));
  }

  void train(std::array<T,N> const & x, T const & target, float learingRate) {
    output.error = target - (*this)(x);
    for (int i=0; i<M; ++i) hidden2.neurons[i].error =  output.w[i]*output.error;
    for (int i=0; i<M; ++i) hidden1.neurons[i].error = 0;
    for (int j=0; j<M; ++j) for (int i=0; i<M; ++i) hidden1.neurons[j].error += hidden2.neurons[i].w[j]*hidden2.neurons[i].error;
    hidden1.updateWeight(learingRate);
    hidden2.updateWeight(learingRate);
    output.updateWeight(learingRate);
  }



  
};

#include <x86intrin.h>
unsigned int taux=0;
inline unsigned long long rdtscp() {
 return __rdtscp(&taux);
}



template<typename Data>
struct Reader {
  explicit Reader(long long iread) : toRead(iread){}
  
  long long operator()(std::vector<Data> & buffer) {
    auto bufSize = buffer.size();
    // random instead of reading
    for ( auto & b : buffer) for (auto & e : b) e=rgen(eng);   // background
    for (int j=0; j<bufSize; j+=4) // one out of 4
      buffer[j][4] = buffer[j][3] = buffer[j][2] = buffer[j][0];  //signal... 
    toRead -= bufSize;
    return toRead;
  }
private:
  long long toRead;
  std::mt19937 eng;
  std::uniform_real_distribution<float> rgen = std::uniform_real_distribution<float>(0.,1.);

};

#include<iostream>
template<int NX, int MNodes>
void go() {

  constexpr long long Nentries = 1024*10000;


  std::mt19937 eng;
  std::uniform_real_distribution<float> rgen(0.,1.);
  std::uniform_real_distribution<float> wgen(-1.,1.);


  NeuNet<float,NX,MNodes> net;

  // random ...
  for (auto & w: net.output.w) w=wgen(eng);
  for (auto & n: net.hidden2.neurons) for (auto & w: n.w) w=wgen(eng);
  for (auto & n: net.hidden1.neurons) for (auto & w: n.w) w=wgen(eng);
  

  // timers
  long long tt=0, tc=0;

  
  using Data = std::array<float,NX>; // one row
  constexpr unsigned int bufSize=1024;
  // "array of struct"
  std::vector<Data> buffer(bufSize);  // an "array" of rows

  // train
  Reader<Data> reader1(Nentries/4);
  while (reader1(buffer)>=0) {
    tt -= rdtscp();
    int ll=4;
    for (auto & b : buffer) {
      float t=0.f;
      if (4==ll) { t=1.f; ll=1;} // signal (see reader)
      else ++ll;
      net.train(b,t,0.02f);
    }
    tt +=rdtscp();
  }

  

  double count=0;
  double pass=0;

  float cut = 0.5;

  
  Reader<Data> reader2(Nentries);
  // classify
  while (reader2(buffer)>=0) {
    tc -= rdtscp();
    for ( auto & b : buffer) {
      if (net(b)>cut) ++pass;
      ++count;
    }
    tc +=rdtscp();

  }
    
  std::cout << "\nInput Size " << NX << " layer size " << MNodes << std::endl;
  std::cout << "total time training " << double(tt)*1.e-9 << std::endl;
  std::cout << "total time classification " << double(tc)*1.e-9 << std::endl;
  std::cout << "final result " << pass/count << std::endl;
}

int main() {

  go<10,14>();
  go<10,7>();

  
  return 0;

}










