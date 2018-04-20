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
void invert (mTest::AMatrix<double,N> * mm) {
  using MX = mTest::AMatrix<double,N>;
  MX & m = *mm;
  
  ROOT::Math::CholeskyDecomp<double,N> decomp(m);
  assert(decomp);
  
  assert(decomp.Invert(m));
 
  
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


  
  using MX = mTest::AMatrix<double,4>;
  MX m;
  genMatrix<MX,double,4>(m);

  std::cout << m(1,1) << std::endl;
 
  auto m_d = cuda::memory::device::make_unique<MX[]>(current_device, 1);
  
  cuda::memory::copy(m_d.get(), &m, sizeof(MX));

   int threadsPerBlock =32;
   int blocksPerGrid = 1;
   cuda::launch(
                invert<4>,
                { blocksPerGrid, threadsPerBlock },
                m_d.get()
		);

   cuda::memory::copy(&m, m_d.get(),sizeof(MX));
		
		
  std::cout << m(1,1) << std::endl;
 
  ROOT::Math::CholeskyDecomp<double,4> decomp(m);
  assert(decomp);
  
  assert(decomp.Invert(m));
  
  std::cout << m(1,1) << std::endl;
  
  
  return 0;
}
