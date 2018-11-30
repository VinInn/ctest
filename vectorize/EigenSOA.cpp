#include <cmath>
#include <Eigen/Core>
#include <tuple>

using Eigen::Vector3f;
using Eigen::Matrix3f;
using v3 = Eigen::Matrix<float,3,1>;
using M = Eigen::Map<Eigen::VectorXf,0,Eigen::InnerStride<> >;
using m = Eigen::Map<Vector3f>;
using n = Eigen::Map<v3>;
using ar = std::tuple<float&,float&,float&>;
void foo(float * __restrict__ b, float * __restrict__ c, float * __restrict__ r) 
{ 
    int N=1024;
  for (int i=0; i<N;++i) {
    M v1(b+i,3,Eigen::InnerStride<>(N));
    M v2(c+i,3,Eigen::InnerStride<>(N));
    r[i] = v1.dot(v2);
  }
}

void bar(Matrix3f k, float * b, float c) 
{ 
    //ar q{b[0],b[64],b[128]};
    for (int i=0; i<64;++i) {
      v3 a1(b[i],b[i+64],b[i+128]);
      a1= k*a1;
      b[i]=a1[0];
      b[i+64]=a1[1];
      b[i+128]=a1[2];
      
    }
}


void dot(float const * __restrict__ b, float * __restrict__ c) 
{ 
    //ar q{b[0],b[64],b[128]};
    for (int i=0; i<64;++i) {
      auto j=i+1024;
      v3 a1(b[i],b[i+64],b[i+128]);
      v3 a2(b[j],b[j+64],b[j+128]);
      c[i] = a1.dot(a2);
    }
}

