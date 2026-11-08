//  nvcc -gencode arch=compute_120,code=sm_120 -O3 --expt-relaxed-constexpr -std=c++23 benchNN.cu -DNT=512 -DNB=4 -DMX=10000
//  /opt/nvidia/nsight-compute/2025.2.1/ncu --metrics sm__cycles_elapsed.avg,sm__sass_thread_inst_executed_op_fp32_pred_on.sum,sm__sass_thread_inst_executed_op_integer_pred_on.sum,sm__sass_thread_inst_executed_pred_on.sum,dram__bytes.sum ./a.out
// ./a.out | grep gtime | cut -d' ' -f6 | tr '\n' ','; echo ']\n,['
// #include "benchSoA.h"
#include <cassert>
#include <cstdint>
#include <cmath>
#include <tuple>
#include <cuda_fp16.h>
#include <iostream>
// #include <crt/device_fp128_functions.h>
#include <bit>

#ifndef HOST_DEVICE_CONSTANT
#ifdef __CUDA_ARCH__
#define HOST_DEVICE_CONSTANT __device__ constexpr
#else
#define HOST_DEVICE_CONSTANT constexpr
#endif
#endif

#ifdef __NVCC__
#define HD_INLINE __device__ __host__ inline
#else
#define HD_INLINE inline
#endif


template<typename T>
 __device__ __forceinline__
T ncm(T const * __restrict__ x) {
#ifdef CONSTM
     if constexpr (std::is_same<__float128, typename std::remove_cv<T>::type>::value)
       return *x;
     else
       return __ldg(x);
#else
     return *x;
#endif
}

template<typename T>
struct SoA {
   using Type = T;
   T * __restrict__ x;
   T * __restrict__ y;
   T * __restrict__ z;

  __device__ __forceinline__
   T xc(int i) const { return ncm(x+i);}
  __device__ __forceinline__
   T yc(int i) const { return ncm(y+i);}
   __device__ __forceinline__
   T zc(int i) const { return ncm(z+i);}

};


template<typename T>
struct U {
   using Float=T;
   HD_INLINE T operator()(T x) { return x;}
};


constexpr uint32_t shift15(uint32_t x) {
    return (x>>15)&(0x00010001);
}
constexpr std::tuple<float,float>  unpack(uint32_t x) {
    uint16_t uj = x&0xFFFF;
    uint16_t ui = (x&0xFFFF0000)>>16;
    return {float(std::bit_cast<int16_t>(ui)),float(std::bit_cast<int16_t>(uj))};
}
constexpr uint32_t pack(float a1, float a2) {
  int16_t i = roundf(a1); int16_t j = roundf(a2);
  uint32_t ui = std::bit_cast<uint16_t>(i);
  uint32_t uj = std::bit_cast<uint16_t>(j);
  return (ui<<16) | uj;
}

template<typename T>
constexpr T assign(double x) { return T(x);}
template<>
constexpr half2 assign(double x) { return half2(x,x);}



#include<random>
template<typename T, typename U>
struct G {
  int * operator()(SoA<T> & x, SoA<U> & y, int n, int nb) { 
    cudaMalloc(&a, 3*n * sizeof(T)); 
    cudaMalloc(&b, 3*n * sizeof(U));
    cudaMalloc(&ind, n * sizeof(int));
    x.x = a; x.y = a+n; x.z = x.y + n;
    y.x = b; y.y = b+n; y.z = y.y + n;

    int l[nb+1];
    T * q = new T[3*n];
    std::random_device rd;  // a seed source for the random number engine
    std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
//    std::uniform_int_distribution<> rint(0,n);
    std::uniform_real_distribution<float> rf(-1.0,1.0);
    std::uniform_int_distribution<> rint16(-32768,32768);

    l[0]=0;
    for (int i=1; i<nb+1;++i) l[i] = l[i-1] + n/nb;  // maybe random in future....
    std::cout << l[0] <<',' << l[1] << ' ' << l[nb]<<std::endl;
    for (int i=0; i<3*n;++i) {
      if constexpr (std::is_floating_point<T>::value || std::is_same<half, typename std::remove_cv<T>::type>::value || std::is_same<__float128, typename std::remove_cv<T>::type>::value)
        q[i] = rf(gen);
      else if constexpr(std::is_same<half2, typename std::remove_cv<T>::type>::value)
         q[i] = half2{rf(gen), rf(gen)};
      else if constexpr(std::is_same<float2, typename std::remove_cv<T>::type>::value)
         q[i] = float2{rf(gen), rf(gen)};
      else if constexpr(std::is_same<uint32_t, typename std::remove_cv<T>::type>::value)
        q[i] = rint16(gen) | (rint16(gen)<<16);
      else
        q[i] = rint16(gen);
    }
    cudaMemcpy(ind, l, (nb+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(a, q, 3*n*sizeof(T),cudaMemcpyHostToDevice);
    delete [] q;
    return ind;
  }
  ~G() {
    cudaFree(a);
    cudaFree(b);
    cudaFree(ind);
  }
  T * a;
  U * b;
  int * ind;
};


// blocks are independent events...
 template<typename T,typename U>
__device__
void bfN2(T inV,  U const outV, int const * __restrict__ ind) {
    const auto cut2 = assign<typename T::Type>(0.016);
    int start = ind[blockIdx.x];
    int end = ind[blockIdx.x+1];
    // if (start==end) printf("empty block %d\n",blockIdx.x);
    int tid = start + threadIdx.x;
    for (auto i=tid; i<end; i+=blockDim.x)  {
      auto x = inV.xc(i);
      auto y = inV.yc(i);
      auto z = inV.zc(i);
      typename U::Type k = i;
      if constexpr(std::is_same<half2, typename std::remove_cv<T>::type>::value) {
        k = (i*2+1)<<16 | i*2;
      }
      for (uint32_t j=start; j<end; j++) {
        auto d2 = (x-inV.xc(j))*(x-inV.xc(j)) + (y-inV.yc(j))*(y-inV.yc(j));
        if constexpr(std::is_same<half2, typename std::remove_cv<T>::type>::value) {
         auto j2 = j*j | (j*j+1)<<16;
         y = inV.zc(j);
         auto m = __hgt2_mask(y,z) & __hgt2_mask(cut2,d2);
         z = __hmax2_nan(__hgt2(cut2,d2)*y,z);
         k = m&j2 | (~m)&k;
         y = __lowhigh2highlow(y);
         m = __hgt2_mask(y,z) & __hgt2_mask(cut2,d2);
         z = __hmax2_nan(__hgt2(cut2,d2)*y,z);
         j2 = j*j+1 | (j*j)<<16;
         k = m&j2 | (~m)&k;
       }else {
        if (inV.zc(j)>z && d2<cut2) { k = j; z=inV.zc(j);}
//        bool doit = inV.zc(j)>z && d2<cut2;
//        k = doit ? j :k; 
//        z = doit ? inV.zc(j) : z;
       }
     }
     outV.x[i] = k;
   }
}

// blocks are independent events...
template<typename T, typename U>
__global__ void benchit(T inV,  U const outV, int const * __restrict__ ind) {

   bfN2(inV,outV,ind);

}




#ifndef NB
#define NB 40
#endif

#ifndef NT
#define NT 128
#endif

#ifndef MX
#define MX 1024
#endif


#include<string>
template<typename G, typename T, typename U, int IB=NB, int IT=NT, int IX=MX>
void doBench(std::string const & fname, int n) {
  constexpr int nB = IB;
  constexpr int nT = IT;
  constexpr int maxIter = IX;
  if (n<=0) n = nB*nT;

  std::cout << fname << " n, nb,nt, mx "  << n << ' ' << nB << ',' << nT << ' ' << maxIter << std::endl;

  T a;  // input
  U b;  // output

  G g;
  int * ind = g(a,b, n, nB);

  benchit<T,U><<<nB,nT,0,0>>>(a, b, ind);
  cudaDeviceSynchronize();
}



int main() {

   int n = MX*NB;

   doBench<G<float,uint32_t>,SoA<float>,SoA<uint32_t>>("f n2",n);
   doBench<G<half2,uint32_t>,SoA<half2>,SoA<uint32_t>>("h n2",n/2);

   return 0;
}
