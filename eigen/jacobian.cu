#include <cmath>
#include <Eigen/Core>


constexpr int stride() { return 5*1024;}
using Matrix5d = Eigen::Matrix<double, 5, 5>;
using Vector5d = Eigen::Matrix<double, 5, 1>;
using MV = Eigen::Map<Vector5d,0,Eigen::InnerStride<stride()> >;
using MD = Eigen::Map<Matrix5d,0,Eigen::Stride<5*stride(),stride()> >;

__global__
void trans(double * __restrict__ pcov,
double * __restrict__ pp, int n) {

  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i>=n) return;
 
   Matrix5d J = Matrix5d::Identity();
  /*
  J << 
  1,0,0,0,0,
  0,1,0,0,0,
  0,0,1,0,0,
  0,0,0,1,0,
  0,0,0,0,1;
  */

  
  MV p(pp+i);
  auto sinTheta2 = 1/(1+p(3)*p(3));
  auto sinTheta = std::sqrt(sinTheta2);
  J(2,2) = -sinTheta/(p(2)*p(2));
  J(2,3) = -sinTheta2*sinTheta*p(3)/p(2);
  

  //auto const & J = pj[i];

  MD cov(pcov+i);
//  auto tmp = cov*J.transpose();
  cov.noalias() = J*(cov*J.transpose()).eval();
}


__global__
void transNaive(double * __restrict__ pcov,
double * __restrict__ pp,
double * __restrict__ ocov,
int n ) {

  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i>=n) return;


  Matrix5d J = Matrix5d::Identity();
  MV p(pp+i);
  auto sinTheta2 = 1/(1+p(3)*p(3));
  auto sinTheta = std::sqrt(sinTheta2);
  J(2,2) = -sinTheta/(p(2)*p(2));
  J(2,3) = -sinTheta2*sinTheta*p(3)/p(2);


  MD cov(pcov+i);
  MD out(ocov+i);
  out.noalias() = J*cov*J.transpose();
}


// usual test stuff
#include <iomanip>
#include <memory>
#include <algorithm>
#include <chrono>
#include<random>


#include<cassert>
#include<iostream>
#include<limits>

#include "cuda/api_wrappers.h"

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

int main() {

  using MX = Matrix5d;
  using VX = Vector5d;
  constexpr auto DIM = 5;

  auto current_device = cuda::device::current::get();
 

  constexpr unsigned int SIZE=4*1024;

  MX mm[stride()];  // just storage in case of SOA
  MX mo[stride()];  // just storage in case of SOA
  VX vv[stride()];  // just storage in case of SOA
  double * __restrict__ pm = (double *)(mm);
  double * __restrict__ po = (double *)(mo);
  double * __restrict__ pv = (double *)(vv);


  for (unsigned int i=0; i<SIZE; ++i) {
    MD m(pm+i);
    genMatrix(m);
    //!<(phi,Tip,pt,cotan(theta)),Zip)
    MV v(pv+i);
    v(2)=v(4)=0;
    v(0)=0.3;
    v(2)=2.;
    v(3)=0.5;
  }


  auto m_d = cuda::memory::device::make_unique<double[]>(current_device, DIM*DIM*stride());
  cuda::memory::copy(m_d.get(), pm, stride()*sizeof(MX));
  auto v_d = cuda::memory::device::make_unique<double[]>(current_device, DIM*stride());
  cuda::memory::copy(v_d.get(), pv, stride()*sizeof(VX));

  auto o_d = cuda::memory::device::make_unique<double[]>(current_device, DIM*DIM*stride());

  int NKK=1000;
  for (int kk=0; kk<NKK; ++kk) {

    int threadsPerBlock =128;
    int blocksPerGrid = SIZE/threadsPerBlock;

      cuda::launch(
           transNaive,
           { blocksPerGrid, threadsPerBlock },
           m_d.get(),v_d.get(),o_d.get(),
           SIZE
           );

    cuda::memory::copy(po, o_d.get(),stride()*sizeof(MX));

      cuda::launch(
           trans,
           { blocksPerGrid, threadsPerBlock },
           m_d.get(),v_d.get(),SIZE
           );

    cuda::memory::copy(pm, m_d.get(),stride()*sizeof(MX));

    int ne=0;
    for (int i=0; i<SIZE*25; ++i) {
     if (std::abs(pm[i]-po[i])>1.e-6) ne++;
    }

    if (0==kk%100) std::cout << "errors " << ne << std::endl;

  }
  return 0;

}
