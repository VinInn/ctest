// /usr/local/cuda/bin/nvcc -gencode arch=compute_75,code=sm_75 -O3 --expt-relaxed-constexpr clock.cu -DCLOCK -DFLOAT=float
// /usr/local/cuda/bin/nvcc -gencode arch=compute_75,code=sm_75 -O3 --expt-relaxed-constexpr -std=c++17 clockMatrix.cu -DFLOAT=float -DTWOF
#include<cstdint>
#include<cmath>
#include<random>
#include<cstdio>
#include<iostream>

#include "Matrix.h"
#include "TwoFloat.h"

#define VERIFY

template <typename M>
inline void verify(M const& m) {
#ifdef VERIFY
#warning "Verify ON"
  int n = M::kRows;
  for (int i = 0; i < n; ++i)
       assert(toSingle(m(i,i))>0);
  //check minors
  auto d = toSingle(m(0, 0)*m(1,1)) - toSingle(m(0, 1)*m(1,0));
  if (d<0) std::cout << "??? " << d << std::endl;
  assert(d > -1.e-8);
  auto d3 = toSingle(m(1, 0)*m(2,1)) - toSingle(m(2, 0)*m(1,1));
  auto d2 = toSingle(m(1, 0)*m(2,2)) - toSingle(m(2, 0)*m(1,2));
  auto d1 = toSingle(m(1, 1)*m(2,2)) - toSingle(m(1, 2)*m(2,1));
  auto dd = toSingle(m(0,0))*d1-toSingle(m(0,1))*d2+toSingle(m(0,2))*d3;
  if (dd<0) std::cout << "??? " << dd << std::endl;
  assert(dd > -1.e-8);
}
#endif

// generate matrices
template <typename M, typename Eng>
void genMatrix(M& m, Eng & eng) {
  // using T = typename std::remove_reference<decltype(m(0, 0))>::type;
  int n = M::kRows;
  std::uniform_real_distribution<float> rgen(0., 1.);

  // generate first diagonal elemets
  for (int i = 0; i < n; ++i) {
    float maxVal = i * 1.e5 / (n - 1) + 1;  // max condition is 10^5
    m(i, i) = maxVal * rgen(eng) + 1.e-10;
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      float v = 0.3f * std::sqrt(toSingle(m(i, i) * m(j, j)));  // this makes the matrix pos defined
      m(i, j) = v * rgen(eng) + 1.e-10;
      if (rgen(eng)<0.5f) m(i, j) = -m(i, j);
      // m(j, i) = m(i, j);
    }
  }
}



using Float = FLOAT;

#if defined(TWOF)
using MM5 = MatrixSym<TwoFloat<Float>,5>;
#else
using MM5 = MatrixSym<Float,5>;
#endif

// Type your code here, or load an example.
__global__ void square(MM5 * array,  int64_t * tt, int64_t * tg, int n) {
     int maxIter = 500000;
#ifdef CLOCK
     __shared__ uint64_t gstart;
#else
     __shared__ uint64_t gstart, gend;
     uint64_t start, end;
#endif
     int tid = blockDim.x * blockIdx.x + threadIdx.x;

     auto m1 = array[tid];
     MM5 m2;

     if (threadIdx.x==0) {
#ifdef CLOCK
      gstart = clock64();
#else
      // Record start time
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gstart));
#endif
     }
     __syncthreads();
#ifdef CLOCK     
    auto s = clock64();
#else    
    // Record start time
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
#endif
    if (tid<n) {
       for (int kk=0; kk<maxIter; ++kk) {
          invert55(m1,m2);
          invert55(m2,m1);
       }
    // Record end time 
#ifdef CLOCK
       tt[tid] = clock64() -s;
#else
   asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end));
   tt[tid] = end - start;
#endif
    }
    __syncthreads();

    if (threadIdx.x==0) {
 #ifdef CLOCK
      tg[blockIdx.x] = clock64() -gstart;
#else
     asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gend));
     tg[blockIdx.x] = gend - gstart;
#endif
   }
   array[tid] = m1;
}

#include<iostream>

#ifndef NB
#define NB 1
#endif

#ifndef NT
#define NT 128
#endif


int main(int argc, char** argv) {

  constexpr int nB = NB;
  constexpr int nT = NT;

   std::cout << "nb,nt "  << nB << ' ' << nT << std::endl;

  constexpr int n = nB*nT;
  MM5 * a;
  int64_t * tt;
  int64_t * tg;

   MM5 m0[n];
   
  cudaMallocManaged(&a, n*sizeof(MM5));
  cudaMallocManaged(&tt, n*sizeof(int64_t));
  cudaMallocManaged(&tg, nB*sizeof(int64_t));

  std::mt19937 eng;
  for (int i=0; i<n; ++i) {
    MM5 w;
    genMatrix(a[i], eng);
    verify(a[i]);
    invert55(a[i],w);
    verify(w);
  }
  for (int i=0; i<n; ++i) m0[i] = a[i];

  for (int i=0; i<n; ++i) tt[i]=0;
  for (int i=0; i<nB; ++i) tg[i]=0;
  square<<<nB,nT,0,0>>>(a,tt,tg,n);
  cudaDeviceSynchronize();

  Float maxOn=0;
  Float maxOff=0;
  int ns = 5;
  for (int i=0; i<n; ++i) {
    auto const & m1 = m0[i];
    auto const & m3 = a[i];
    verify(m3);
    for (int i=0; i<ns; ++i)
#if defined(TWOF)
     maxOn = std::max(maxOn,std::abs( ((m3(i,i)-m1(i,i))/m1(i,i)).hi() ));
#else
      maxOn = std::max(maxOn,std::abs( (m3(i,i)-m1(i,i))/m1(i,i) ));
#endif
    for (int i = 0; i < ns; ++i) {
      for (int j = 0; j < i; ++j) {
#if defined(TWOF)
         maxOff = std::max(maxOn,std::abs( ((m3(i,j)-m1(i,j))/m1(i,j)).hi() ));
#else
         maxOff = std::max(maxOn,std::abs( (m3(i,j)-m1(i,j))/m1(i,j) ));
#endif
      }
    }
  }
  std::cout << maxOn << ' ' << maxOff << std::endl;

  for (int i=0; i<n; ++i) std::cout << tt[i] <<  ' ';
  std::cout << '\n' << std::endl;
  std::cout << "gtime ";
  for (int i=0; i<nB; ++i) std::cout << tg[i] <<  ' ';
  std::cout << '\n' << std::endl;

  cudaFree(a);
  cudaFree(tt);
  cudaFree(tg);

  return 0;
}
