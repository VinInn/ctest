// /usr/local/cuda/bin/nvcc -gencode arch=compute_75,code=sm_75 -O3 --expt-relaxed-constexpr -std=c++23 clockMatrix.cu -DNT=512 -DNB=4
#pragma once
#include<cstdint>
#include<cmath>
#include<random>
#include<cstdio>
#include<iostream>
#include<limits>


template<typename F>
__host__ __device__ constexpr void init(F & f) {}


template<typename F, typename T, typename U=T>
__global__ void clockit(T * outV,  U const * inV, int64_t * tt, int64_t * tg, int n,  int maxIter) {
     __shared__  long long ostart, lstart, lend;
     __shared__  unsigned long long  gstart, gend;

     int tid = blockDim.x * blockIdx.x + threadIdx.x;
     __shared__ F f;
     auto m1 = inV[tid];
     /*volatile*/ T m2=0;

     if (threadIdx.x==0) {
      init(f);
      ostart = clock64();
      gstart = std::numeric_limits<unsigned long long>::max();
      gend=0;  lstart=std::numeric_limits<long long>::max(); lend=0;
     }
     __syncthreads();

    if (tid<n) {
      unsigned long long ss;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ss));
      atomicMin(&gstart,ss);
      auto s = clock64();
      atomicMin(&lstart,s);
       for (int kk=0; kk<maxIter; ++kk) {
          m2 = f(m1+U(m2*T(.1e-12)));
       }
       // Record end time 
      auto e = clock64();
      tt[tid] = e - s;
      atomicMax(&lend,e);
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ss));
      atomicMax(&gend,ss);
    }
    __syncthreads();

    if (threadIdx.x==0) {
      tg[blockIdx.x] = clock64() -ostart;
      tg[blockIdx.x+gridDim.x] =  lend - lstart;
      tg[blockIdx.x+2*gridDim.x] =  gend - gstart;

    }

    outV[tid]=m2;
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
void doClock(std::string const & fname="") {
  constexpr int nB = NB;
  constexpr int nT = NT;
  constexpr int maxIter = MX;

  std::cout << "nb,nt "  << nB << ' ' << nT << std::endl;

  constexpr int n = nB*nT;
  U * a;
  T * b;
  int64_t * tt;
  int64_t * tg;

   
  cudaMallocManaged(&a, n*sizeof(U));
  cudaMallocManaged(&b, n*sizeof(T));
  cudaMallocManaged(&tt, n*sizeof(int64_t));
  cudaMallocManaged(&tg, 3*nB*sizeof(int64_t));

  G g;
  for (int i=0; i<n; ++i) a[i] = g(i);

  for (int i=0; i<n; ++i) tt[i]=0;
  for (int i=0; i<nB; ++i) tg[i]=0;
  clockit<F,T,U><<<nB,nT,0,0>>>(b, a, tt,tg,n, maxIter);
  cudaDeviceSynchronize();

  std::cout << fname << "(" <<a[nT-1] <<") = "<< b[nT-1] << std::endl;

#ifdef THTIME
  for (int i=0; i<n; ++i) std::cout << tt[i] <<  ' ';
  std::cout << '\n' << std::endl;
#endif
  std::cout << "gtime ";
  for (int i=0; i<nB; ++i) 
     std::cout << '(' << tg[i] << ' ' << tg[i+nB] <<  ' ' << tg[i+nB+nB] << ") ";
  std::cout << '\n' << std::endl;

  cudaFree(a);
  cudaFree(b);
  cudaFree(tt);
  cudaFree(tg);
}
