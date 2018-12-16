// nvcc -O3 CholeskyDecomp_t.cu -Icuda-api-wrappers/src/ --expt-relaxed-constexpr -gencode arch=compute_61,code=sm_61 --compiler-options="-Ofast -march=native"
// add -DDOPROF to run  nvprof --metrics all

#include "choleskyInversion.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <iomanip>
#include <memory>
#include <algorithm>
#include <chrono>
#include<random>


#include<cassert>
#include<iostream>
#include<limits>

constexpr int stride() { return 5*1024;}
template<int DIM>
using MXN = Eigen::Matrix<double,DIM,DIM>;
template<int DIM>
using MapMX = Eigen::Map<MXN<DIM>, 0, Eigen::Stride<DIM*stride(),stride()> >;

// generate matrices
template<class M>
void genMatrix(M  & m ) {
  using T = typename std::remove_reference<decltype(m(0,0))>::type;
  int n = M::ColsAtCompileTime;
  std::mt19937 eng;
  // std::mt19937 eng2;
  std::uniform_real_distribution<T> rgen(0.,1.);
  
  // generate first diagonal elemets
  for (int i = 0; i < n; ++i) {
    double maxVal = i*10000/(n-1) + 1;  // max condition is 10^4
    m(i,i) = maxVal*rgen(eng);
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      double v = 0.3*std::sqrt( m(i,i) * m(j,j) ); // this makes the matrix pos defined
      m(i,j) = v*rgen(eng);
      m(j,i) = m(i,j);
    }
  }
}


template<int N>
void go(bool soa) {

  constexpr unsigned int DIM = N;
  using MX =  MXN<DIM>;
  std::cout << "testing Matrix of dimension " << DIM << " size " << sizeof(MX) << std::endl;


  auto start = std::chrono::high_resolution_clock::now();
  auto delta = start - start;
  auto delta1 = delta;
  auto delta2 = delta;

  
  constexpr unsigned int SIZE=4*1024;
  
  MX mm[stride()];  // just storage in case of SOA
  double * __restrict__ p = (double *)(mm);

  if (soa) {
    for (unsigned int i=0; i<SIZE; ++i) {
      MapMX<N> m(p+i);
      genMatrix(m);
    }
  }else{  
    for ( auto & m : mm) 
      genMatrix(m);
  }

  std::cout << mm[SIZE/2](1,1) << std::endl;

  if (soa)
    for (unsigned int i=0; i<SIZE; ++i) {
      MapMX<N> m(p+i);
      choleskyInversion::invert(m,m);
      choleskyInversion::invert(m,m);
    }
  else
    for ( auto & m : mm) {
      choleskyInversion::invert(m,m);
      choleskyInversion::invert(m,m);
    }

  std::cout << mm[SIZE/2](1,1) << std::endl;


  constexpr int NKK = 
#ifdef DOPROF
    2;
#else
    1000;
#endif
  for (int kk=0; kk<NKK; ++kk) {
  
    delta2 -= (std::chrono::high_resolution_clock::now()-start);
    if (soa)
      #pragma GCC ivdep
      for (unsigned int i=0; i<SIZE; ++i) {
	MapMX<N> m(p+i);
	choleskyInversion::invert(m,m);
      }
    else
      #pragma GCC ivdep
      for ( auto & m : mm) {
	choleskyInversion::invert(m,m);
      }
    
    delta2 += (std::chrono::high_resolution_clock::now()-start);

  }
  
  std::cout << mm[SIZE/2](1,1) << std::endl; 
  
  double DNNK = NKK;
  std::cout <<"x86 computation took "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(delta2).count()/DNNK  << ' '
	    << " ms" << std::endl;

}

int main() { 

  go<2>(false);
  go<4>(false);
  go<5>(false);
  go<6>(false);

  go<2>(true);
  go<4>(true);
  go<5>(true);
  go<6>(true);

  // go<10>();
  return 0;
}
