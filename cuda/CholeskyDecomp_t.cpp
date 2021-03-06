#include "CholeskyDecomp.h"


#include <iomanip>
#include <memory>
#include <algorithm>
#include <chrono>
#include<random>


#include<cassert>
#include<iostream>
#include<limits>



// generate matrices
template<class M, class T, int N>
void genMatrix(M  & m ) {
  
  std::mt19937 eng;
  // std::mt19937 eng2;
  std::uniform_real_distribution<T> rgen(0.,1.);
  
  // generate first diagonal elemets
  for (int i = 0; i < N; ++i) {
    double maxVal = i*10000/(N-1) + 1;  // max condition is 10^4
    m(i,i) = maxVal*rgen(eng);
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < i; ++j) {
      double v = 0.3*std::sqrt( m(i,i) * m(j,j) ); // this makes the matrix pos defined
      m(i,j) = v*rgen(eng);
      m(j,i) = m(i,j);
    }
  }
}


template<int N> 
void  go() {

  
  constexpr int SIZE=1024;

  std::cout << "inverting " << SIZE << " matrices of rank " << N << std::endl;   

  using MX = mTest::AMatrix<double,N>;
  MX mm[SIZE];
  for ( auto & m : mm) 
    genMatrix<MX,double,N>(m);


  std::cout << mm[SIZE/2](1,1) << std::endl;

  for ( auto & m : mm) {
    ROOT::Math::CholeskyDecomp<double,N> decomp(m);
    assert(decomp);
    assert(decomp.Invert(m));
  }
  
  std::cout << mm[SIZE/2](1,1) << std::endl;
}

int main() {

  go<4>();
  go<5>();
  go<10>();
  
  return 0;
}
