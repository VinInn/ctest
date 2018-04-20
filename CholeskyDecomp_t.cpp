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


int main() {
  
  using MX = mTest::AMatrix<double,4>;
  MX m;
  genMatrix<MX,double,4>(m);


  std::cout << m(1,1) << std::endl;
 
  ROOT::Math::CholeskyDecomp<double,4> decomp(m);
  assert(decomp);
  
  assert(decomp.Invert(m));
  
  std::cout << m(1,1) << std::endl;
  
  return 0;
}
