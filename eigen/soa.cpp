#include <cmath>
#include <Eigen/Core>

using m34 = Eigen::Matrix<float,3,4>;
using MD = Eigen::Map<m34,0,Eigen::Stride<3*1024,1024> >;


#include<iostream>

template<typename M>
void print(M const & m) {
   std::cout << m.cols() << std::endl;
   std::cout << m << std::endl;
}

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
    m1.col(i) << 20+i,200+i,2000+i;  
  std::cout << m1 << std::endl;

  std::cout << std::endl;
  print(m1);

  std::cout << data[0] << ' ' << data[1] << std::endl;
  std::cout << data[1024] << ' ' << data[1024+1] << std::endl;

  std::cout << std::endl;
  print(m1.block(0,0,2,m1.cols()));

  
  return 0;

}
