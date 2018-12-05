#include <cmath>
#include <Eigen/Core>

using m34 = Eigen::Matrix<float,3,4>;
using MD = Eigen::Map<m34,0,Eigen::Stride<3*1024,1024> >;



#include<iostream>
int main() {

  m34 a;
  for (auto i=0; i<4; ++i) 
    a.col(i) << 10+i,100+i,1000+i;

  std::cout << a << std::endl;

  float data[12*1024];

  MD m(data,3,4);
  for (auto i=0; i<4; ++i)
    m.col(i) << 10+i,100+i,1000+i;
  std::cout << m << std::endl;

  MD m1(data+1,3,4);
  for (auto i=0; i<4; ++i)
    m1.col(i) << 10+i,100+i,1000+i;  
  std::cout << m1 << std::endl;

  std::cout << data[0] << ' ' << data[1] << std::endl;
  std::cout << data[1024] << ' ' << data[1024+1] << std::endl;

  
  return 0;

}
