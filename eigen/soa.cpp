#include <cmath>
#include <Eigen/Core>
#include <cassert>


using m34 = Eigen::Matrix<float,3,4>;
using MD = Eigen::Map<m34,0,Eigen::Stride<m34::RowsAtCompileTime*1024,1024> >;

using V15 = Eigen::Matrix<float,15,1>;


template<typename M, int S>
struct VectorSOA {
  using Scalar = typename M::Scalar; 
  using Map = Eigen::Map<M, 0, Eigen::InnerStride<S>>;
  using CMap = Eigen::Map<const M, 0, Eigen::InnerStride<S>>;


  constexpr Map operator()(uint32_t i)  { return Map(data+i);}
  constexpr CMap operator()(uint32_t i) const { return CMap(data+i);}

  Scalar data[S*M::RowsAtCompileTime];
};


template<typename M, int S>
struct MatrixSOA {
  using Scalar = typename M::Scalar; 
  using Map = Eigen::Map<M, 0, Eigen::Stride<M::RowsAtCompileTime*S,S> >;
  using CMap = Eigen::Map<const M, 0, Eigen::Stride<M::RowsAtCompileTime*S,S> >;

  constexpr Map operator()(uint32_t i)  { return Map(data+i);}
  constexpr CMap operator()(uint32_t i) const { return CMap(data+i);}

  Scalar data[S*M::RowsAtCompileTime*M::ColsAtCompileTime];
};


#include<iostream>

template<typename M>
void print(M const & m) {
   std::cout << m.cols() << std::endl;
   std::cout << m << std::endl;
}

template<typename M>
void printProp() {

  using F = typename M::Scalar;
  std::cout << "scalar " << typeid(F).name() << std::endl;
  std::cout << "rs/cs " << M::RowsAtCompileTime << '/' << M::ColsAtCompileTime << std::endl;
  std::cout << "size " << sizeof(M) <<std::endl;
  
};


int main() {

  printProp<Eigen::Vector3d>();
  printProp<V15>();
  printProp<m34>();
  printProp<MD>();

  VectorSOA<Eigen::Vector3d,1024> mv3;
  MatrixSOA<Eigen::Vector3d,1024> mm3;

  std::cout << "soa sizes " << sizeof(mv3) << ' ' << sizeof(mm3) << std::endl;
  
  for (int i=0; i<1024; ++i) {
    mv3(i) << i*10,i*10+1,i*10+2;
    mm3(i) << i*10,i*10+1,i*10+2;
  }

  for (int i=0; i<1024; ++i) {
    assert(mv3(i)(1)==mm3(i)(1));
  }

  for (int i=0; i<3*1024; ++i) {
    assert(mv3.data[i]==mm3.data[i]);
  }
  
  
  m34 a;
  for (auto i=0; i<4; ++i) 
    a.col(i) << 10+i,100+i,1000+i;

  std::cout << a << std::endl;

  float data[12*1024];

  MD m(data); // ,3,4);
  for (auto i=0; i<4; ++i)
    m.col(i) << 10+i,100+i,1000+i;
  std::cout << m << std::endl;

  MD m1(data+1); // ,3,4);
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
