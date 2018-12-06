#include <cmath>
#include <Eigen/Core>

#include<iostream>


int main() {

  int n=2;
  Eigen::Matrix4f VcsF;
              VcsF << 1,12,12,14,
                    11,2,23,24, 
                    12,23,3,34,
                    14,24,34,4;

  Eigen::Matrix4f Vcs;

  Vcs.triangularView<Eigen::Upper>() = VcsF;


  std::cout <<Vcs << std::endl;

  Eigen::Matrix2f Vcs_[2][2];

    Vcs_[0][0] = Vcs.block(0, 0, n, n);
    Vcs_[0][1] = Vcs.block(0, n, n, n);
    Vcs_[1][1] = Vcs.block(n, n, n, n);
    Vcs_[1][0] = Vcs_[0][1].transpose();
 

   std::cout << Vcs_[0][0] << std::endl;
   std::cout << Vcs_[0][1] << std::endl;
   std::cout << Vcs_[1][1] << std::endl;
   std::cout << Vcs_[1][0] << std::endl;

};
