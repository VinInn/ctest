#include <cmath>
#include <Eigen/Core>


constexpr int stride() { return 5*1024;}
using Matrix5d = Eigen::Matrix<double, 5, 5>;
using Vector5d = Eigen::Matrix<double, 5, 1>;
using MV = Eigen::Map<Vector5d,0,Eigen::InnerStride<stride()> >;
using MD = Eigen::Map<Matrix5d,0,Eigen::Stride<5*stride(),stride()> >;

void trans(double * __restrict__ pcov,
double * __restrict__ pp, int n) {

  #pragma GCC ivdep
  #pragma clang loop vectorize(enable) interleave(enable)
  for (int i=0; i<n; ++i) {
 
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
  Matrix5d tmp = cov*J.transpose();
  cov.noalias() = J*tmp;
}
}

void transNaive(double * __restrict__ pcov,
double * __restrict__ pp,
double * __restrict__ ocov,
int n ) {

  #pragma GCC ivdep
  #pragma clang loop vectorize(enable) interleave(enable)
  for (int i=0; i<n; ++i) {


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


  auto start = std::chrono::high_resolution_clock::now();
  auto delta = start - start;
  auto delta1 = delta;
  auto delta2 = delta;


  using MX = Matrix5d;
  using VX = Vector5d;
  constexpr auto DIM = 5;

 

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



  int NKK=1000;
  for (int kk=0; kk<NKK; ++kk) {

    delta2 -= (std::chrono::high_resolution_clock::now()-start);
    transNaive(pm,pv,po,SIZE);
    delta2 += (std::chrono::high_resolution_clock::now()-start);
    delta1 -= (std::chrono::high_resolution_clock::now()-start);
    trans(pm,pv,SIZE);
    delta1 += (std::chrono::high_resolution_clock::now()-start);

    int ne=0;
    for (int i=0; i<SIZE*25; ++i) {
     if (std::abs(pm[i]-po[i])>1.e-6) ne++;
    }

    if (0==kk%100) std::cout << "errors " << ne << std::endl;

  }

  double DNNK = NKK;
  std::cout <<"x86 computation took (opt,naive) "
        << std::chrono::duration_cast<std::chrono::milliseconds>(delta1).count()/DNNK  << ' '
        << std::chrono::duration_cast<std::chrono::milliseconds>(delta2).count()/DNNK  << ' '
        << " ms" << std::endl;
  return 0;

}
