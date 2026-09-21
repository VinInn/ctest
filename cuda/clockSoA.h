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


template<typename F, typename T, typename U=T>
__global__ void clockit(T outV,  U const inV, int64_t * tt, int64_t * tg, int n,  int maxIter) {
     __shared__  long long ostart, lstart, lend;
     __shared__  unsigned long long  gstart, gend;

     int tid = blockDim.x * blockIdx.x + threadIdx.x;
     if (tid>=n) return;
     __shared__ F f;

     if (threadIdx.x==0) {
      init(f);
      ostart = clock64();
      gstart = std::numeric_limits<unsigned long long>::max();
      gend=0;  lstart=std::numeric_limits<long long>::max(); lend=0;
     }
     __syncthreads();

    unsigned long long ss;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ss));
    atomicMin(&gstart,ss);
    auto s = clock64();
    atomicMin(&lstart,s);

    for (auto j=threadIdx.x; j<n; j+=blockDim.x)  {
       for (int kk=0; kk<maxIter; ++kk) {
           f(outV,inV, j, kk);
       }
    }

    // Record end time 
    auto e = clock64();
    tt[tid] = e - s;
    atomicMax(&lend,e);
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ss));
    atomicMax(&gend,ss);
    __syncthreads();

    if (threadIdx.x==0) {
      tg[blockIdx.x] = clock64() -ostart;
      tg[blockIdx.x+gridDim.x] =  lend - lstart;
      tg[blockIdx.x+2*gridDim.x] =  gend - gstart;

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
template<typename G, typename F, typename T, typename U=T>
void doClockSoA(std::string const & fname="", int n=0 ) {
  constexpr int nB = NB;
  constexpr int nT = NT;
  constexpr int maxIter = MX;

  std::cout << "nb,nt "  << nB << ' ' << nT << std::endl;

  if (n<=0) n = nB*nT;
  U a;  // input
  T b;  // output
  int64_t * tt;
  int64_t * tg;

  cudaMallocManaged(&tt, n*sizeof(int64_t));
  cudaMallocManaged(&tg, 3*nB*sizeof(int64_t));

  G g;
  g(a,b,n);

  for (int i=0; i<n; ++i) tt[i]=0;
  for (int i=0; i<nB; ++i) tg[i]=0;
  clockit<F,T,U><<<nB,nT,0,0>>>(b, a, tt,tg,n, maxIter);
  cudaDeviceSynchronize();

//   std::cout << fname << "(" <<a[nT-1] <<") = "<< b[nT-1] << std::endl;

#ifdef THTIME
  for (int i=0; i<n; ++i) std::cout << tt[i] <<  ' ';
  std::cout << '\n' << std::endl;
#endif
  std::cout << "gtime ";
  for (int i=0; i<nB; ++i) 
     std::cout << '(' << tg[i] << ' ' << tg[i+nB] <<  ' ' << tg[i+nB+nB] << ") ";
  std::cout << '\n' << std::endl;

  cudaFree(tt);
  cudaFree(tg);
}
