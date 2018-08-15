// nvcc -O3 CholeskyDecomp_t.cu -Icuda-api-wrappers/src/ --expt-relaxed-constexpr -gencode arch=compute_61,code=sm_61 --compiler-options="-Ofast -march=native"
// add -DDOPROF to run  nvprof --metrics all
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
void invert (mTest::AMatrix<double,N> * mm, unsigned int n) {

  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i>=n) return;

  auto & m = mm[i];
  
  ROOT::Math::CholeskyDecomp<double,N> decomp(m);
  assert(decomp);
  
  assert(decomp.Invert(m));
 
}

template<int N>
__global__
void invertSeq (mTest::AMatrix<double,N> * mm, unsigned int n) {

  if (threadIdx.x!=0) return;
  auto first = blockIdx.x*blockDim.x;
  auto last = std::min(first+blockDim.x,n);
  
  for (auto i=first; i<last; ++i) {
    auto & m = mm[i];
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


template<int N>
void go() {

  constexpr unsigned int DIM = N;
  using MX = mTest::AMatrix<double,DIM>;

  std::cout << "testing Matrix of dimension " << DIM << " size " << sizeof(MX) << std::endl;
  using MX = mTest::AMatrix<double,DIM>;

  auto start = std::chrono::high_resolution_clock::now();
  auto delta = start - start;
  auto delta1 = delta;
  auto delta2 = delta;

  
  if (cuda::device::count() == 0) {
	std::cerr << "No CUDA devices on this system" << "\n";
	exit(EXIT_FAILURE);
  }

  auto current_device = cuda::device::current::get(); 

  constexpr unsigned int SIZE=1024;
  
  MX mm[SIZE];
  for ( auto & m : mm) 
    genMatrix<MX,double,DIM>(m);
  

  std::cout << mm[SIZE/2](1,1) << std::endl;

  for ( auto & m : mm) {
   ROOT::Math::CholeskyDecomp<double,DIM> decomp(m);
   assert(decomp);
   assert(decomp.Invert(m));
   ROOT::Math::CholeskyDecomp<double,DIM> decomp2(m);
   assert(decomp2);
   assert(decomp2.Invert(m));
  }

  std::cout << mm[SIZE/2](1,1) << std::endl;

   auto m_d = cuda::memory::device::make_unique<MX[]>(current_device, SIZE);
   cuda::memory::copy(m_d.get(), &mm, SIZE*sizeof(MX));


  constexpr int NKK = 
#ifdef DOPROF
    2;
#else
    1000;
#endif
  for (int kk=0; kk<NKK; ++kk) {
  
    // auto m_d = cuda::memory::device::make_unique<MX[]>(current_device, SIZE);
    
    // cuda::memory::copy(m_d.get(), &mm, SIZE*sizeof(MX));
    
    int threadsPerBlock =128;
    int blocksPerGrid = SIZE/threadsPerBlock;
    
    delta -= (std::chrono::high_resolution_clock::now()-start);
    
    cuda::launch(
		 invert<DIM>,
		 { blocksPerGrid, threadsPerBlock },
		 m_d.get(),SIZE
		 );
    
    cuda::memory::copy(&mm, m_d.get(),SIZE*sizeof(MX));
    delta += (std::chrono::high_resolution_clock::now()-start);
    
    if (0==kk) std::cout << mm[SIZE/2](1,1) << std::endl;
    
    
    delta1 -= (std::chrono::high_resolution_clock::now()-start);

#ifndef DOPROF
     cuda::launch(
		 invertSeq<DIM>,
		 { blocksPerGrid, threadsPerBlock },
		 m_d.get(),SIZE
		 );
    
    cuda::memory::copy(&mm, m_d.get(),SIZE*sizeof(MX));
#endif
    delta1 += (std::chrono::high_resolution_clock::now()-start);
    
    if (0==kk) std::cout << mm[SIZE/2](1,1) << std::endl;
    
  
    delta2 -= (std::chrono::high_resolution_clock::now()-start);
    for ( auto & m : mm) {
      ROOT::Math::CholeskyDecomp<double,DIM> decomp(m);
      assert(decomp);
      assert(decomp.Invert(m));
    }
    delta2 += (std::chrono::high_resolution_clock::now()-start);

  }
  
  std::cout << mm[SIZE/2](1,1) << std::endl; 
  
  double DNNK = NKK;
  std::cout <<"cuda/cudaSeq/x86 computation took "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()/DNNK << ' '
	    << std::chrono::duration_cast<std::chrono::milliseconds>(delta1).count()/DNNK  << ' '
	    << std::chrono::duration_cast<std::chrono::milliseconds>(delta2).count()/DNNK  << ' '
	    << " ms" << std::endl;

}

int main() { 

  go<2>();
  go<4>();
  go<5>();

  go<10>();
  return 0;
}
