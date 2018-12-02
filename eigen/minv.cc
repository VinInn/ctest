#include <cmath>
#include <Eigen/Core>
#include <tuple>

#include "invert55.h"

using v5 = Eigen::Matrix<float,5,1>;
using m5 = Eigen::Matrix<float,5,5>;
using M = Eigen::Map<m5,0,Eigen::Stride<5*1024,1024> >;
using V = Eigen::Map<v5,0,Eigen::InnerStride<1024> >;

using md5 = Eigen::Matrix<double,5,5>;
using MD = Eigen::Map<md5,0,Eigen::Stride<5*1024,1024> >;


void inv(double * __restrict__ b, double * __restrict__ r)
{
  #pragma GCC ivdep
  for (int i=0; i<256;++i) {
    MD src(b+i,5,5);
    MD dst(r+i,5,5);
    choleksyInvert55(src,dst);
    symmetrize(dst); // if needed...
  }
}

void inv(float * __restrict__ b, float * __restrict__ r)
{
  #pragma GCC ivdep
  for (int i=0; i<256;++i) {
    M src(b+i,5,5);
    M dst(r+i,5,5);
    choleksyInvert55(src,dst);
    symmetrize(dst); //    if needed...
  }
}
