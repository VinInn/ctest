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

#include <vector>
#include <array>



#include "choleskyShift.h"
#include "choleskyLLT.h"

using Float = float; // double; // float;

using DynStride = Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>;
constexpr int stride() { return 5*1024;}
template<int DIM>
using MXN = Eigen::Matrix<Float,DIM,DIM>;
template<int DIM>
using MapMX = Eigen::Map<MXN<DIM>, 0, Eigen::Stride<DIM*stride(),stride()> >;
template<int DIM>
using DynMapMX = Eigen::Map<MXN<DIM>, 0, DynStride >;
template<int DIM>
using MDN = Eigen::Matrix<Float,Eigen::Dynamic,Eigen::Dynamic, 0, DIM,DIM>;


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
    Float maxVal = i*10000/(n-1) + 1;  // max condition is 10^4
    m(i,i) = maxVal*rgen(eng);
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      Float v = 0.3*std::sqrt( m(i,i) * m(j,j) ); // this makes the matrix pos defined
      m(i,j) = v*rgen(eng);
      m(j,i) = m(i,j);
    }
  }
}



int main() {

 constexpr int RANK=8;

 auto start = std::chrono::high_resolution_clock::now();
 auto delta1 = start - start;
 auto delta2 = delta1;
 auto delta3 = delta1;
 long long n1=0, n2=0, n3=0;

for (int im=0; im<100; ++im) {
 MXN<RANK> om;
 if (im==4000) {
   om <<
   7.06342e-05,  2.40066e-05, -6.30805e-06, -3.05303e-06, -4.46876e-06,  -2.0358e-05, -3.09587e-05,   6.5016e-05,
   2.40066e-05,  0.000154432, -6.36697e-06, -5.33037e-06, -4.85508e-06, -2.25869e-05,  2.62186e-05,  1.17919e-05,
   -6.30805e-06, -6.36697e-06,  7.74101e-05,  2.89917e-06,  2.53991e-05,  7.16527e-05,  5.80385e-05, -9.67691e-06,
   -3.05303e-06, -5.33037e-06,  2.89917e-06,    6.096e-05,  3.61575e-05, -6.50032e-06, -1.52493e-05, -4.20482e-06,
   -4.46876e-06, -4.85508e-06,  2.53991e-05,  3.61575e-05,  8.60409e-05,   3.2623e-06, -8.04242e-06, -6.16502e-06,
   -2.0358e-05, -2.25869e-05,  7.16527e-05, -6.50032e-06,   3.2623e-06,  0.000292254,  0.000387729, -2.89142e-05,
   -3.09587e-05,  2.62186e-05,  5.80385e-05, -1.52493e-05, -8.04242e-06,  0.000387729,  0.000775488,  -4.7374e-05,
    6.5016e-05,  1.17919e-05, -9.67691e-06, -4.20482e-06, -6.16502e-06, -2.89142e-05,  -4.7374e-05,   0.00014075;
 } else if (im==400) {
    om <<
 8.29042e-06,  3.30608e-06,  6.29528-07,  1.66786e-07, -5.23755e-08, -1.12201e-07, -2.04543e-07, -2.84626e-07,
 3.30608e-06,  1.09074e-05,  3.28107e-06,   9.3326e-07,   1.3443e-07, -8.67955e-08, -2.56688e-07, -3.57188e-07,
 6.29528e-07,  3.28107e-06,  1.09091e-05,  5.08012e-06,  1.06032e-06,  -1.9255e-08, -1.72311e-07, -3.86656e-07,
 1.66786e-07,   9.3326e-07,  5.08012e-06,  1.65243e-05,  6.07789e-06,  1.57804e-07, -9.33208e-08, -2.61259e-07,
-5.23755e-08,   1.3443e-07,  1.06032e-06,  6.07789e-06,  1.81996e-05,  1.27573e-06,  3.84391e-07,  2.37103e-08,
-1.12201e-07, -8.67955e-08,  -1.9255e-08,  1.57804e-07,  1.27573e-06,  4.30836e-06,  2.48321e-06,  1.16214e-06,
-2.04543e-07, -2.56688e-07, -1.72311e-07, -9.33208e-08,  3.84391e-07,  2.48321e-06,  8.66693e-06,  5.43413e-06,
-2.84626e-07, -3.57188e-07, -3.86656e-07, -2.61259e-07,  2.37103e-08,  1.16214e-06,  5.43413e-06,  1.68775e-05;
 } else
   genMatrix(om);

 int p = RANK;
 if(im==4) std::cout << om << std::endl << std::endl;

 auto lu = om.llt();
 if (lu.info() != Eigen::Success) {
    std::cout << "numerical problem in first lu for " << im << std::endl;
    continue;
 }

 auto ldlt = om;
 choleskyLLT(ldlt,ldlt.rows());

 auto ori = ldlt; // a copy
 if(im==4) std::cout << "ldlt\n"<< ori << std::endl << std::endl;

 for (int k=0; k<p-1; ++k) {
 for (int l=k+1; l<p; ++l) {
 auto m = om;

 /*
 choleskyShiftUp(ldlt,k,l);
 if(im==4&&k==2&&l==4) std::cout << "ldlt up\n" << ldlt << std::endl << std::endl;
 choleskyShiftDown(ldlt,k,l);
 if(im==4&&k==2&&l==4) std::cout << "ldlt down\n" << ldlt << std::endl << std::endl;
 */

 if(im==4&&k==2&&l==4) std::cout << "shift up" << std::endl;
 delta2 -= (std::chrono::high_resolution_clock::now() -start);
 choleskyShiftUp(ldlt,k,l); 
 delta2 += (std::chrono::high_resolution_clock::now() -start);
 n2++;
 if(im==4&&k==2&&l==4) std::cout << ldlt << std::endl << std::endl;

 if(im==4&&k==2&&l==4) std::cout << "now  shift" << std::endl;
 
 for (int i=k; i<l; ++i) {
   m.col(i).swap(m.col(i+1));
   m.row(i).swap(m.row(i+1));
 }
 auto r = m;

 if(im==4&&k==2&&l==4) std::cout << m << std::endl << std::endl;
   delta1 -= (std::chrono::high_resolution_clock::now() -start);
   choleskyLLT(r,r.rows());
   delta1 += (std::chrono::high_resolution_clock::now() -start);
   n1++;

 if(im==4&&k==2&&l==4) std::cout << r << std::endl << std::endl;


   bool ok=true;
  {
    auto d = (r-ldlt).eval();
    for (int j=0;j<p; ++j)
    for (int i=0;i<=j; ++i)
       ok&=std::abs(d(j,i)/r(j,i))<0.001;
    if(!ok) {
      std::cout << "mess in shift up " << im << ' ' << k << ' ' << l << std::endl;
      std::cout << r << std::endl<< std::endl;
      std::cout << ldlt << std::endl<< std::endl;
      std::cout << ori << std::endl<< std::endl;
      std::cout << lu.matrixLLT() << std::endl<< std::endl;
      std::cout << om << std::endl<< std::endl;
      abort();
    }
  }

 if(im==4&&k==2&&l==4) std::cout << "shift down" <<    std::endl;
 delta3 -= (std::chrono::high_resolution_clock::now() -start);
 choleskyShiftDown(ldlt,k,l);
 delta3 += (std::chrono::high_resolution_clock::now() -start);
 n3++;
 if(im==4&&k==2&&l==4) std::cout << ldlt << std::endl << std::endl;

 if (ok) {
    auto d = (ori-ldlt).eval();
    for (int j=0;j<p; ++j)
    for (int i=0;i<=j; ++i)
       ok&=std::abs(d(j,i)/ori(j,i))<0.001;
    if(!ok) { 
      std::cout << "mess in shift down " << im << ' ' << k << ' ' << l << std::endl;
      std::cout << ori << std::endl<< std::endl;
      std::cout << ldlt << std::endl<< std::endl;
      abort();
    }
  }

  if(im==4&&k==2&&l==4) {
  // just for me to make sure is L and not U
  r(0,0) = 45;
  std::cout << r(0,0) << std::endl;
  std::cout << r(1,0) << std::endl;
  std::cout << r << std::endl << std::endl;
  }

}} // loops on k&l  
} // loop on 100 matrices

  std::cout << "ldlt  "  << std::chrono::duration_cast<std::chrono::nanoseconds>(delta1).count()/double(n1) << std::endl; 
  std::cout << "up   "  << std::chrono::duration_cast<std::chrono::nanoseconds>(delta2).count()/double(n2) << std::endl;
  std::cout << "down "  << std::chrono::duration_cast<std::chrono::nanoseconds>(delta3).count()/double(n3) << std::endl;

  return 0;
}


