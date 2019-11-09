#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Cholesky>

#include <iomanip>
#include <memory>
#include <algorithm>
#include <chrono>
#include<random>


#include<cassert>
#include<iostream>
#include<limits>


#include "choleskyShift.h"

using DynStride = Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>;
constexpr int stride() { return 5*1024;}
template<int DIM>
using MXN = Eigen::Matrix<double,DIM,DIM>;
template<int DIM>
using MapMX = Eigen::Map<MXN<DIM>, 0, Eigen::Stride<DIM*stride(),stride()> >;
template<int DIM>
using DynMapMX = Eigen::Map<MXN<DIM>, 0, DynStride >;


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

 constexpr int RANK=6;

 MXN<RANK> m;
 genMatrix(m);

 int p = RANK;
 std::cout << m << std::endl << std::endl;

 auto lu = m.llt();
 std::cout << lu.matrixLLT() << std::endl << std::endl;

 std::cout << "shift up 2,4" << std::endl;
 auto & ru = const_cast<MXN<RANK>&>(lu.matrixLLT());
 choleskyShiftUp(ru,2,4); 
 std::cout << ru << std::endl << std::endl;

 std::cout << "now  shift" << std::endl;
 m.col(2).swap(m.col(3));
 m.row(2).swap(m.row(3));
 m.col(4).swap(m.col(3));
 m.row(4).swap(m.row(3));

 std::cout << m << std::endl << std::endl;
 auto lus = m.llt();
 std::cout << lus.matrixLLT() << std::endl << std::endl;

  auto & r = const_cast<MXN<RANK>&>(lus.matrixLLT());

  auto d = (r-ru).eval();
  for (int j=0;j<p; ++j)
  for (int i=0;i<=j; ++i)
    assert(std::abs(d(j,i))<0.001);


 std::cout << "shift down  2,4" <<    std::endl;
 choleskyShiftDown(ru,2,4);
 std::cout << ru << std::endl << std::endl;



  // just for me to make sure is L and not U
  r(0,0) = 45;
  std::cout << r(0,0) << std::endl;
  std::cout << r(1,0) << std::endl;
  std::cout << lus.matrixLLT() << std::endl << std::endl;

  return 0;
}


