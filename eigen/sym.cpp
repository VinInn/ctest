//  /usr/bin/c++ -O3 -std=c++17 -march=native sym.cpp -I/Users/innocent/eigen -I/Users/innocent/RealStuff/ctest/floatPrec
#include <cmath>
#include <Eigen/Core>

#include "TwoFloat.h"

#include<iostream>

using FF = TwoFloat<float>;

int main() {

  int n=2;
  Eigen::Matrix<FF,4,4> VcsF;
              VcsF << 1.,12.,12.,14.,
                    11.,2.,23.,24., 
                    12.,23.,3.,34.,
                    14.,24.,34.,4.;

  Eigen::Matrix<FF,2,2> j; j<< 1.,-1.,0.,0.5;

  Eigen::Matrix<FF,1,2> v; v<<1.,-1.;
  Eigen::Matrix<FF,2,1> w; w<<1.,-1.;


  Eigen::Matrix<FF,4,4> Vcs;

  std::cout << Eigen::Matrix4f::RowsAtCompileTime << ' ' << Eigen::Matrix4f::ColsAtCompileTime << std::endl;
  std::cout << Eigen::Vector4f::RowsAtCompileTime << ' ' << Eigen::Vector4f::ColsAtCompileTime << std::endl;

  Vcs.triangularView<Eigen::Upper>() = VcsF;


  std::cout <<Vcs << std::endl;

  Eigen::Matrix<FF,2,2> Vcs_[2][2];

    Vcs_[0][0] = Vcs.block(0, 0, n, n);
    Vcs_[0][1] = Vcs.block(0, n, n, n);
    Vcs_[1][1] = Vcs.block(n, n, n, n);
    Vcs_[1][0] = Vcs_[0][1].transpose();
 

   std::cout << Vcs_[0][0] << std::endl;
   std::cout << Vcs_[0][1] << std::endl;
   std::cout << Vcs_[1][1] << std::endl;
   std::cout << Vcs_[1][0] << std::endl;

   std::cout << std::endl;


   Eigen::Matrix<FF,2,2> C[3][3];
   std::cout << std::endl;
   C[0][0] = Vcs_[0][0].selfadjointView<Eigen::Upper>();
   std::cout << C[0][0] << std::endl;

   std::cout << std::endl;

   C[1][0].triangularView<Eigen::Upper>() = j*C[0][0].selfadjointView<Eigen::Upper>()*j.transpose();
   std::cout << toSingle(C[1][0]) << std::endl;


   C[1][1].triangularView<Eigen::Upper>() = (Vcs_[0][0].array()*Vcs_[0][0].array()).matrix().triangularView<Eigen::Upper>();
   std::cout << toSingle(C[1][1]) << std::endl;

   std::cout << std::endl;


  Eigen::Matrix<FF,1,1> c;
  c = v*C[0][0].selfadjointView<Eigen::Upper>()*v.transpose();
  std::cout << c << std::endl;


  c = w.transpose()*C[0][0].selfadjointView<Eigen::Upper>()*w;
  std::cout << c << std::endl;


  Eigen::Matrix<FF,2,4> q = VcsF.block(0, 0, 2, 4);

  Eigen::Matrix<FF,2,2> qq = q*q.transpose();
  std::cout << qq<< std::endl;


  Eigen::Matrix<FF,2,2> qqq;
  qqq.triangularView<Eigen::Upper>() = q*q.transpose();
  std::cout << qqq<< std::endl;

  Eigen::Matrix<FF,2,2> qqqq;
  qqqq.triangularView<Eigen::Upper>() = q*q.transpose();
  std::cout << qqqq<< std::endl;

   return 0;
}
