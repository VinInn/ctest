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

 constexpr int RANK=8;


for (int im=0; im<100; ++im) {
 MXN<RANK> om;
 genMatrix(om);

 int p = RANK;
 if(im==4) std::cout << om << std::endl << std::endl;

 auto lu = om.llt();

 auto ori = lu.matrixLLT(); // a copy
 if(im==4) std::cout << ori << std::endl << std::endl;

 for (int k=0; k<p-1; ++k) {
 for (int l=k+1; l<p; ++l) {
 auto m = om;
 if(im==4&&k==2&&l==4) std::cout << "shift up" << std::endl;
 auto & ru = const_cast<MXN<RANK>&>(lu.matrixLLT());
 choleskyShiftUp(ru,k,l); 
 if(im==4&&k==2&&l==4) std::cout << ru << std::endl << std::endl;

 if(im==4&&k==2&&l==4) std::cout << "now  shift" << std::endl;
 
 for (int i=k; i<l; ++i) {
   m.col(i).swap(m.col(i+1));
   m.row(i).swap(m.row(i+1));
 }

 if(im==4&&k==2&&l==4) std::cout << m << std::endl << std::endl;
 auto lus = m.llt();
 if(im==4&&k==2&&l==4) std::cout << lus.matrixLLT() << std::endl << std::endl;

  auto & r = const_cast<MXN<RANK>&>(lus.matrixLLT());

   bool ok=true;
  {
    auto d = (r-ru).eval();
    for (int j=0;j<p; ++j)
    for (int i=0;i<=j; ++i)
       ok&=std::abs(d(j,i))<0.001;
    if(!ok) {
      std::cout << "mess in shift up " << im << ' ' << k << ' ' << l << std::endl;
      std::cout << r << std::endl<< std::endl;
      std::cout << ru << std::endl<< std::endl;
      abort();
    }
  }

 if(im==4&&k==2&&l==4) std::cout << "shift down" <<    std::endl;
 choleskyShiftDown(ru,k,l);
 if(im==4&&k==2&&l==4) std::cout << ru << std::endl << std::endl;

 if (ok) {
    auto d = (ori-ru).eval();
    for (int j=0;j<p; ++j)
    for (int i=0;i<=j; ++i)
       ok&=std::abs(d(j,i))<0.001;
    if(!ok) { 
      std::cout << "mess in shift down " << im << ' ' << k << ' ' << l << std::endl;
      std::cout << ori << std::endl<< std::endl;
      std::cout << ru << std::endl<< std::endl;
      abort();
    }
  }

  if(im==4&&k==2&&l==4) {
  // just for me to make sure is L and not U
  r(0,0) = 45;
  std::cout << r(0,0) << std::endl;
  std::cout << r(1,0) << std::endl;
  std::cout << lus.matrixLLT() << std::endl << std::endl;
  }

}} // loops on k&l  
} // loop on 100 matrices
  return 0;
}


