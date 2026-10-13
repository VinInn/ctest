// /usr/local/cuda/bin/nvcc -gencode arch=compute_75,code=sm_75 -O3 --expt-relaxed-constexpr -std=c++23 clockMatrix.cu -DNT=512 -DNB=4
#pragma once
#include<cstdint>
#include<cmath>
#include<random>
#include<cstdio>
#include<iostream>
#include<limits>
#include<type_traits>

template<typename F>
__host__ __device__ constexpr void init(F & f) {}

template<typename F, typename T, typename U>
__global__ void benchit(T outV,  U const inV, int const * __restrict__ ind, int n, int maxIter) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
     if (tid>=n) return;
     __shared__ F f;

    for (auto j=tid; j<n; j+=gridDim.x * blockDim.x)  {
       for (int kk=0; kk<maxIter; ++kk) {
           f(outV,inV, ind, j, kk, n);
       }
    }
}

#include<iostream>

#ifndef NB
#define NB 1
#endif

#ifndef NT
#define NT 128
#endif

#ifndef MX
#define MX 5000
#endif

#include<string>
template<typename G, typename F, typename T, typename U=T, int IB=NB, int IT=NT, int IX=MX>
void doBenchSoA(std::string const & fname="", int n=0) {
  constexpr int nB = IB;
  constexpr int nT = IT;
  constexpr int maxIter = IX;
  if (n<=0) n = nB*nT;

  std::cout << fname << " n, nb,nt, mx "  << n << ' ' << nB << ',' << nT << ' ' << maxIter << std::endl;

  U a;  // input
  T b;  // output

  G g;
  int * ind = g(a,b, n);

  benchit<F,T,U><<<nB,nT,0,0>>>(b, a, ind, n, maxIter);
  cudaDeviceSynchronize();

//   std::cout << fname << "(" <<a[nT-1] <<") = "<< b[nT-1] << std::endl;
}
