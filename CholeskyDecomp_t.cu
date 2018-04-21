#include "CholeskyDecomp.h"


#include <iomanip>
#include <memory>
#include <algorithm>
#include <chrono>
#include<random>


#include<cassert>
#include<iostream>
#include<limits>

#include "cuda/api_wrappers.h"

template<int N>
__global__
void invert (mTest::AMatrix<double,N> * mm, int n) {

  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i>=n) return;

  using MX = mTest::AMatrix<double,N>;
  MX & m = mm[i];
  
  ROOT::Math::CholeskyDecomp<double,N> decomp(m);
  assert(decomp);
  
  assert(decomp.Invert(m));
 
}

template<int N>
__global__
void invertSeq (mTest::AMatrix<double,N> * mm, int n) {

  if (threadIdx.x!=0) return;
  auto first = blockIdx.x*blockDim.x;
  auto last = std::mmin(first+blockDim.x,n);
  
  for (i=first; i<last; ++i) {
    using MX = mTest::AMatrix<double,N>;
    MX & m = mm[i];
    ROOT::Math::CholeskyDecomp<double,N> decomp(m);
    assert(decomp);
    assert(decomp.Invert(m));
  }
}



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

  if (cuda::device::count() == 0) {
	std::cerr << "No CUDA devices on this system" << "\n";
	exit(EXIT_FAILURE);
  }

  auto current_device = cuda::device::current::get(); 

  constexpr int SIZE=1024;
  
  using MX = mTest::AMatrix<double,4>;
  MX mm[SIZE];
  for ( auto & m : mm) 
    genMatrix<MX,double,4>(m);
  

  std::cout << mm[SIZE/2](1,1) << std::endl;

  
  auto m_d = cuda::memory::device::make_unique<MX[]>(current_device, SIZE);
  
  cuda::memory::copy(m_d.get(), &m, SIZE*sizeof(MX));

  int threadsPerBlock =128;
  int blocksPerGrid = SIZE/threadsPerBlock;

  cuda::launch(
	       invert<4>,
	       { blocksPerGrid, threadsPerBlock },
	       m_d.get(),SIZE,
	       );

  cuda::memory::copy(&m, m_d.get(),SIZE*sizeof(MX));
  
  std::cout << mm[SIZE/2](1,1) << std::endl;
  
  cuda::launch(
	       invertSeq<4>,
	       { blocksPerGrid, threadsPerBlock },
	       m_d.get(),SIZE,
	       );

  cuda::memory::copy(&m, m_d.get(),SIZE*sizeof(MX));
  
  std::cout << mm[SIZE/2](1,1) << std::endl;
  
  
  
  for ( auto & m : mm) {
    ROOT::Math::CholeskyDecomp<double,4> decomp(m);
    assert(decomp);
    assert(decomp.Invert(m));
  }
  
  std::cout << m[SIZE/2](1,1) << std::endl; 
  
 
  
   return 0;
}
