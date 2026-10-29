//  nvcc -gencode arch=compute_75,code=sm_75 -O3 --expt-relaxed-constexpr -std=c++23 benchNN.cu -DNT=512 -DNB=4 -DMX=10000
//  /opt/nvidia/nsight-compute/2025.2.1/ncu --metrics sm__cycles_elapsed.avg,sm__sass_thread_inst_executed_op_fp32_pred_on.sum,sm__sass_thread_inst_executed_op_integer_pred_on.sum,sm__sass_thread_inst_executed_pred_on.sum,dram__bytes.sum ./a.out
// ./a.out | grep gtime | cut -d' ' -f6 | tr '\n' ','; echo ']\n,['
#include "benchSoA.h"
#include <cassert>
#include <cmath>
#include <cuda_fp16.h>
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
template<typename T>
struct G {
  int * operator()(SoA<T> & x, SoA<T> & y, int n, int nb) { 
    cudaMalloc(&a, 3*n * sizeof(T)); 
    cudaMalloc(&b, 3*n * sizeof(T));
    cudaMalloc(&ind, n * sizeof(int));
    x.x = a; x.y = a+n; x.z = x.y + n;
    y.x = b; y.y = b+n; y.z = y.y + n;

    int l[nb+1];
    T * q = new T[3*n];
    std::random_device rd;  // a seed source for the random number engine
    std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> rint(0,n);
    std::uniform_real_distribution<float> rf(-1.0,1.0);
    std::uniform_int_distribution<> rint16(-32768,32768);

    l[0]=0;
    for (int i=1; i<nb+1;++i) l[i] = l[i-1] + n/nb;  // maybe random in future....

    for (int i=0; i<3*n;++i) {
      if constexpr (std::is_floating_point<T>::value || std::is_same<half, typename std::remove_cv<T>::type>::value || std::is_same<__float128, typename std::remove_cv<T>::type>::value)
        q[i] = rf(gen);
      else if constexpr(std::is_same<half2, typename std::remove_cv<T>::type>::value)
         q[i] = half2{rf(gen), rf(gen)};
      else if constexpr(std::is_same<float2, typename std::remove_cv<T>::type>::value)
         q[i] = float2{rf(gen), rf(gen)};
      else
        q[i] = rint16(gen);
    }
    cudaMemcpy(ind, l, nb*sizeof(int),cudaMemcpyHostToDevice);
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
  T * b;
  int * ind;
};


// blocks are independent events...
template<typename T, typename U>
__global__ void benchit(T outV,  U const inV, int const * __restrict__ ind) {
    int start = ind[blockIdx.x] + threadIdx.x;
    int end = ind[blockIdx.x+1];

     __shared__ int c;

    for (auto j=start; j<end; j+=blockDim.x)  {
        c++;
    }
    outV.x[start]=c; 
}




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
template<typename G, typename T, typename U=T, int IB=NB, int IT=NT, int IX=MX>
void doBench(std::string const & fname, int n) {
  constexpr int nB = IB;
  constexpr int nT = IT;
  constexpr int maxIter = IX;
  if (n<=0) n = nB*nT;

  std::cout << fname << " n, nb,nt, mx "  << n << ' ' << nB << ',' << nT << ' ' << maxIter << std::endl;

  U a;  // input
  T b;  // output

  G g;
  int * ind = g(a,b, n, nB);

  benchit<T,U><<<nB,nT,0,0>>>(b, a, ind);
  cudaDeviceSynchronize();
}



int main() {

   int n = 24*1024*NB;

   doBench<G<float>,SoA<float>,SoA<float>>("f n2",n);

   return 0;
}
