#include <cmath>
#include <Eigen/Core>
#include <tuple>

using v5 = Eigen::Matrix<float,5,1>;
using m5 = Eigen::Matrix<float,5,5>;
using M = Eigen::Map<m5,0,Eigen::Stride<5*1024,1024> >;
using V = Eigen::Map<v5,0,Eigen::InnerStride<1024> >;

__global__
void bar(float * __restrict__ c,
    float * __restrict__ j,
    float * __restrict__ r, int ss) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>ss) return;

    M cm(c+i,5,5);
    M jm(j+i,5,5);
    M rm(r+i,5,5);

    rm.noalias() = jm*cm*jm.transpose();
}

__global__
void foo(m5 * __restrict__ c,
    m5 * __restrict__ j,
    m5 * __restrict__ r, int ss) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>ss) return;
    r[i].noalias() = j[i]*c[i]*j[i].transpose();
}


int main() {

  float *c, *j, *r;
  cudaMalloc(&c, 1024*sizeof(m5));
  cudaMalloc(&j, 1024*sizeof(m5));
  cudaMalloc(&r, 1024*sizeof(m5));

  bar<<<1,1024>>>(c,j,r,512);

  foo<<<1,1024>>>((m5*)c,(m5*)j,(m5*)r,512);

  cudaDeviceSynchronize();



}
