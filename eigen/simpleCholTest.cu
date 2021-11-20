
template<typename M1, typename M2, int N=M2::ColsAtCompileTime>
// __host__ __device__
constexpr
void invertNN(M1 const & src, M2 & dst) {
    using T = typename M2::Scalar;

  T a[N][N];
  for (int i=0; i<N; ++i) {
    a[i][i]=src(i,i);
    for (int j=i+1; j<N; ++j)
      // a[i][j] =
      a[j][i] = src(i,j);
  }


  for (int j=0; j<N; ++j) {
    a[j][j]  =  T(1.) / a[j][j];
    int jp1  =  j+1;
    for (int l=jp1; l<N; ++l) {
      a[j][l]  =  a[j][j]*a[l][j];
      T s1 =  -a[l][jp1];
      for (int i=0; i<jp1;++i)
        s1+= a[l][i]*a[i][jp1];
      a[l][jp1]  =  -s1;
    }
  }

  if constexpr (N==1)  { dst(0,0) = a[0][0]; return; }
  a[0][1]  =  -a[0][1];
  a[1][0]  =   a[0][1]*a[1][1];
  for (int j=2; j<N; ++j) {
    int jm1 = j - 1;
    for (int k=0; k<jm1; ++k) {
      T s31  =  a[k][j];
      for (int i=k; i<jm1; ++i)
        s31  += a[k][i+1]*a[i+1][j];
      a[k][j]  =  -s31;
      a[j][k]  =  -s31*a[j][j];
    }
    a[jm1][j]  =  -a[jm1][j];
    a[j][jm1]  =   a[jm1][j]*a[j][j];
  }

  int j=0;
  while (j<N-1) {
    T s33  =  a[j][j];
    for (int i=j+1; i<N; ++i)
      s33  +=  a[j][i]*a[i][j];
    dst(j,j) = s33;

    ++j;
    for (int k = 0; k<j; ++k) {
      T s32  = 0;
      for (int i=j; i<N; ++i)
        s32  +=  a[k][i]*a[i][j];
      dst(k,j) = dst(j,k) = s32;
    }
  }
  dst(j,j)=a[j][j];

}



#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <iomanip>
#include <memory>
#include <algorithm>
#include <chrono>
#include<random>


#include<cassert>
#include<iostream>
#include<limits>


template<typename M, int N>
__global__
void invert (M * mm, int n) {

  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i>=n) return;

  auto & m = mm[i];

  printf("before %d %f %f %f\n",N,m(0,0),m(1,0),m(1,1));  

  invertNN(m,m);

  printf("before %d %f %f %f\n",N,m(0,0),m(1,0),m(1,1));

}

// generate matrices
template<class M>
void genMatrix(M  & m ) {
  using T = typename std::remove_reference<decltype(m(0,0))>::type;
  int n = M::ColsAtCompileTime;
  std::mt19937 eng;
  // std::mt19937 eng2;
  std::uniform_real_distribution<T> rgen(0.,1.);

  // generate first diagonal elemets
  for (int i = 0; i < n; ++i) {
    double maxVal = i*10000/(n-1) + 1;  // max condition is 10^4
    m(i,i) = maxVal*rgen(eng);
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      double v = 0.3*std::sqrt( m(i,i) * m(j,j) ); // this makes the matrix pos defined
      m(i,j) = v*rgen(eng);
      m(j,i) = m(i,j);
    }
  }
}


template<int DIM>
using MXN = Eigen::Matrix<double,DIM,DIM>;



int main() {

  constexpr int DIM = 2;


  using M = MXN<DIM>;

  M m;

  genMatrix(m);

  double * d;
  cudaMalloc(&d,sizeof(M));
  cudaMemcpy(d,&m,sizeof(M),cudaMemcpyHostToDevice);
  invert<M,DIM><<<1,1>>>((M*)d,1);
  cudaDeviceSynchronize();

  return 0;

}
