// Type your code here, or load an example.
#include <cmath>
#include <cstdint>
#include <Eigen/Core>
#include <Eigen/Geometry>

using V3 = Eigen::Vector3f;
using DynStride = Eigen::InnerStride<Eigen::Dynamic>;
using CStride = Eigen::InnerStride<1024>;
using MapV3 =  Eigen::Map<V3,0, DynStride>;
using CMapV3 =  Eigen::Map<const V3,0,  DynStride>;


#include<iostream>
inline void cross(float const * __restrict__ v1, float const * __restrict__ v2, float * __restrict__ v3, int n) {
  for (int tid=0; tid<n; ++tid) {
    CMapV3 m1(v1+tid, V3::RowsAtCompileTime, V3::ColsAtCompileTime, DynStride(n));
    CMapV3 m2(v2+tid, V3::RowsAtCompileTime, V3::ColsAtCompileTime, DynStride(n));
    MapV3 m3(v3+tid, V3::RowsAtCompileTime, V3::ColsAtCompileTime, DynStride(n));
    m3 = m1.cross(m2);
  }
}


int main() {

  float v[3*1024];

  for (int i=0; i<1024; ++i) {
    v[i] =1;
    v[i+1024] =2;
    v[i+2*1024] =3;
  }

  std::cout << V3::RowsAtCompileTime << ' ' << V3::ColsAtCompileTime << std::endl;
  int k=0;
  CMapV3 m(v+k, V3::RowsAtCompileTime, V3::ColsAtCompileTime, DynStride(1024));
  std::cout << m[0] << ' ' << m[1] << ' ' << m[2] << std::endl;


  k=1023;
  new (&m) CMapV3(v+k, V3::RowsAtCompileTime, V3::ColsAtCompileTime, DynStride(1024));
  std::cout << m[0] << ' ' << m[1] << ' ' << m[2] << std::endl;

  return 0;
}

