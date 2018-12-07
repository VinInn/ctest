#include <cmath>
#include <Eigen/Core>

#include<iostream>


int main() {

  using H4f = Eigen::HermitianMatrix<float,4>;
  using H2f = Eigen::HermitianMatrix<float,2>;

  int n=2;
  H4f Vcs;
              Vcs << 1,12,13,14,
                    12,2,23,24, 
                    13,23,3,34,
                    14,24,34,4;

  Eigen::Matrix2f j; j<< 1,-1,0,0.5;

  Eigen::Matrix<float,1,2> v; v<<1,-1;
  Eigen::Matrix<float,2,1> w; w<<1,-1;


  Eigen::Matrix4f VcsF;

  VcsF = Vcs;


  std::cout <<Vcs << std::endl;
  
/*
    Eigen::Matrix2f tmp = VcsF.block(0, 0, n, n);
    H2f Vcs_00 = tmp;
    Eigen::Matrix2f Vcs_01 = VcsF.block(0, n, n, n);
    H2f Vcs_11 = VcsF.block(n, n, n, n);
    Eigen::Matrix2f Vcs_10 = Vcs_01.transpose();
 

   std::cout << Vcs_00 << std::endl;
   std::cout << Vcs_01 << std::endl;
   std::cout << Vcs_11 << std::endl;
   std::cout << Vcs_10 << std::endl;

   std::cout << std::endl;
   
  */ 

  
   H2f C[3][3];
   std::cout << std::endl;

   C[0][0] << 1,12,12,2;

   // C[0][0] = Vcs_00;
   std::cout << C[0][0] << std::endl;

   std::cout << std::endl;

   C[1][0] = j*C[0][0]*j.transpose();
   std::cout << C[1][0] << std::endl;

/*
   C[1][1] = (Vcs_[0][0].array()*Vcs_[0][0].array()).matrix();
   std::cout << C[1][1] << std::endl;

   std::cout << std::endl;


  Eigen::Matrix<float,1,1> c;
  c = v*C[0][0]*v.transpose();
  std::cout << c << std::endl;


  c = w.transpose()*C[0][0].selfadjointView<Eigen::Upper>()*w;
  std::cout << c << std::endl;


  Eigen::Matrix<float,2,4> q = VcsF.block(0, 0, 2, 4);

  H2 qq = q*q.transpose();
  std::cout << qq<< std::endl;


  Eigen::Matrix<float,2,2> qqq;
  qqq  = q*q.transpose();
  std::cout << qqq<< std::endl;

  Eigen::Matrix<float,2,2> qqqq;
  qqqq.triangularView<Eigen::Upper>() = q*q.transpose();
  std::cout << qqqq<< std::endl;
  */
   return 0;
}
