//  nvcc -gencode arch=compute_120,code=sm_120 -O3 --expt-relaxed-constexpr -std=c++23 benchNN.cu -DNT=512 -DNB=4 -DMX=10000
//  /opt/nvidia/nsight-compute/2025.2.1/ncu --metrics sm__cycles_elapsed.avg,sm__sass_thread_inst_executed_op_fp32_pred_on.sum,sm__sass_thread_inst_executed_op_integer_pred_on.sum,sm__sass_thread_inst_executed_pred_on.sum,dram__bytes.sum ./a.out
// ./a.out | grep gtime | cut -d' ' -f6 | tr '\n' ','; echo ']\n,['
// #include "benchSoA.h"

#include "prefixScanN.h"

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
template<typename T, typename U, bool TJ>
struct G {
  static const bool TW = TJ;
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
    std::uniform_real_distribution<float> rf(-M_PI,M_PI);
    std::uniform_int_distribution<> rint16(-32768,32768);
    float scale = TJ ? 0.125  : 1.;
    float xc1 = 0;
    float yc1 = 0.;
    float xc2 = 0.;
    float yc2 = 0.;

    auto cj = [&]() { return TJ ? std::clamp(rf(gen),float(-M_PI+0.4),float(M_PI-0.4)) : 0.; };

    auto xr = [&](){return scale*rf(gen) + (rint16(gen)>0 ? xc1 : xc2);};
    auto yr = [&](){return scale*rf(gen) + (rint16(gen)>0 ? yc1 : yc2);};
    auto zr = [&](){return fabsf(rf(gen));};

    l[0]=0;
    auto nt = n/nb;
    for (int i=1; i<nb+1;++i) l[i] = l[i-1] + n/nb;  // maybe random in future....
    std::cout << l[0] <<',' << l[1] << ' ' << l[nb]<<std::endl;
    for (int i=0; i<n;++i) {
      if (0==(i%nt)) {
         xc1= cj();
         xc2= cj();
      }
      if constexpr (std::is_floating_point<T>::value || std::is_same<half, typename std::remove_cv<T>::type>::value)
        q[i] = xr();
      else if constexpr(std::is_same<half2, typename std::remove_cv<T>::type>::value)
         q[i] = half2{xr(), xr()};
    }
    for (int i=n; i<2*n;++i) {
         if (0==(i%nt)) {
         yc1= cj();
         yc2= cj();
      }
      if constexpr (std::is_floating_point<T>::value || std::is_same<half, typename std::remove_cv<T>::type>::value)
        q[i] = yr();
      else if constexpr(std::is_same<half2, typename std::remove_cv<T>::type>::value)
         q[i] = half2{yr(), yr()};
    }
    for (int i=2*n; i<3*n;++i) {
      if constexpr (std::is_floating_point<T>::value || std::is_same<half, typename std::remove_cv<T>::type>::value)
        q[i] = zr();
      else if constexpr(std::is_same<half2, typename std::remove_cv<T>::type>::value)
         q[i] = half2{zr(), zr()};
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

template<typename T>
__device__ __host__
uint32_t toInd(T x) { return x;}
template<>
__device__
uint32_t toInd<half2>(half2 x) { return (toInd(__high2half(x)) <<16) | toInd(__low2half(x));;}


// blocks are independent events...
 template<typename T,typename U>
__device__
void tile1D(T inV,  U const outV, int const * __restrict__ ind) {
    using F = typename T::Type;
    const auto cut2 = assign<F>(0.016);
    // constexpr uint32_t one = (1<<16) | 1;
    const auto eight = assign<F>(8);
    const auto nine = assign<F>(9);
    const auto invD = assign<F>(1./0.4);
    const auto f16 = assign<F>(16.);
    const auto f0 = assign<F>(0.);

    int start = ind[blockIdx.x];
    int end = ind[blockIdx.x+1];
    // if (start==end) printf("empty block %d\n",blockIdx.x);
    int tid = start + threadIdx.x;

    __shared__ int32_t tiles[16+3];   // guards and total
    if(threadIdx.x<19) tiles[threadIdx.x]=0;
    __syncthreads();

    assert(end-start<=1024);
    __shared__ uint32_t ws[6*1024];  // later dynamic
//    uint16_t * sws  =  (uint16_t*)ws; 
    F * x = (F*)(ws+1024);
    F * y = (F*)(ws+2*1024);
    F * z = (F*)(ws+3*1024);
    uint32_t * it = (uint32_t *)(ws+4*1024);
    uint32_t * oi = (uint32_t *)(ws+5*1024);

    // count
    for (auto i=tid; i<end; i+=blockDim.x)  {
      auto l = i-start;
      auto x = inV.xc(i);
      x = x*invD + nine;  // clamp?
      assert(x>=F(0));
      assert(x<F(17));
      atomicAdd(tiles+toInd(x),1);
      ws[l] = toInd(x);
      assert(ws[l]>0);
      assert(ws[l]<17);
    }
    __syncthreads();
    prefixScanN<int32_t,18>(tiles+1);
    __syncthreads();
   if(threadIdx.x==0) {
     assert(tiles[0]==0);
     assert(tiles[16]==tiles[17]); // guard
     assert(tiles[16]==tiles[18]); // tot
   }
   auto ntot = tiles[18];
   assert(ntot == end-start);
    // fill
    for (auto i=tid; i<end; i+=blockDim.x)  {
      auto l = i-start;
      auto k = atomicSub(tiles+ws[l],1);
      // if(k>=ntot) printf ("??? %d, %d %d %d\n",l,k,ws[l],ntot);
      assert(k>0);
      assert(k<=ntot);
      --k;
      x[k] = inV.xc(i);
      y[k] = inV.yc(i);
      z[k] = inV.zc(i);
      it[k] = ws[l];
      oi[k] = i;
    }
    __syncthreads();
    if(threadIdx.x==0) assert(tiles[1]==0);
    if(threadIdx.x<17) assert(tiles[threadIdx.x+1]>=tiles[threadIdx.x]);
    assert(ntot == tiles[18]);
    if constexpr(std::is_same<half, typename std::remove_cv<T>::type>::value) {
    half2 * x2 = (half2 *)x;
    half2 * y2 = (half2 *)y;
    half2 * z2 = (half2 *)z;
    for (auto i=threadIdx.x; i<ntot/2; i+=blockDim.x)  {
      typename U::Type k = i | (i+1)<<16;
      auto z = z2[i];
      for (uint32_t j=tiles[it[i]-1]; j<tiles[it[i]+3]; j+=2) {
       auto j2 = j/2;
       auto xl = x2[j2];
       auto yl = y2[j2];
       auto zl = z2[j2];
       auto d2 = (xl-x[i])*(xl-x[i]) + (yl-y[i])*(yl-y[i]);
       auto jj = j2*2 | (j2*2+1)<<16;
       auto m = __hgt2_mask(zl,z) & __hgt2_mask(cut2,d2);
       z = __hmax2_nan(__hgt2(cut2,d2)*zl,z);
       k = m&jj | (~m)&k;
       xl = __lowhigh2highlow(xl);
       xl = __lowhigh2highlow(yl);
       zl = __lowhigh2highlow(zl);
       d2 = (xl-x[i])*(xl-x[i]) + (yl-y[i])*(yl-y[i]);
       jj = j2*2+1 | (j2*2)<<16;
       m = __hgt2_mask(zl,z) & __hgt2_mask(cut2,d2);
       z = __hmax2_nan(__hgt2(cut2,d2)*zl,z);
       k = m&jj | (~m)&k;

      }
      outV.x[start+2*i] = oi[k&0xffff];
      outV.x[start+2*i+1] = oi[k>>16];
    }
    }else {
    for (auto i=threadIdx.x; i<ntot; i+=blockDim.x)  {
      typename U::Type k = i;
      auto zb = z[i];
      for (uint32_t j=tiles[it[i]-1]; j<tiles[it[i]+3]; j++) {
       auto d2 = (x[j]-x[i])*(x[j]-x[i]) + (y[j]-y[i])*(y[j]-y[i]);
       if (z[j]>zb && d2<cut2) { k = j; zb=z[j];}
      }
      outV.x[start+i] = oi[k];
    }
    }
}


// blocks are independent events...
 template<typename T,typename U>
__device__
void bfN2(T const inV,  U outV, int const * __restrict__ ind) {
    using F = typename T::Type;
    const auto cut2 = assign<F>(0.016);
    int start = ind[blockIdx.x];
    int end = ind[blockIdx.x+1];
    int ntot = end-start;
#ifdef INSHARED
    __shared__ F ws[3*1024];
    T lV; lV.x = ws; lV.y = ws+1024; lV.z=ws+2*1024;
    for (auto i=threadIdx.x; i<ntot; i+=blockDim.x)  {    
       auto j=  start+i;
       ws[i] = inV.xc(j);
       ws[i+1024] = inV.yc(j);
       ws[i+2*1024] = inV.zc(j);
    }
#else
    T lV; lV.x = inV.x+start; lV.y = inV.y+start; lV.z=inV.z+start;
#endif
    __syncthreads();
    // if (start==end) printf("empty block %d\n",blockIdx.x);
    // int tid = start + threadIdx.x;
    for (auto i=threadIdx.x; i<ntot; i+=blockDim.x)  {
    // for (auto i=tid; i<end; i+=blockDim.x)  {
      auto x = lV.xc(i);
      auto y = lV.yc(i);
      auto z = lV.zc(i);
      typename U::Type k = i;
      if constexpr(std::is_same<half2, typename std::remove_cv<T>::type>::value) {
        k = (i*2+1)<<16 | i*2;
      }
      for (uint32_t j=0; j<ntot; j++) {
        if constexpr(std::is_same<half2, typename std::remove_cv<T>::type>::value) {
         auto j2 = j*2 | (j*2+1)<<16;
         auto xl = lV.xc(j);
         auto yl = lV.yc(j);
         auto zl = lV.zc(j);
         auto d2 = (xl-x)*(xl-x) + (yl-y)*(yl-y);
         auto m = __hgt2_mask(zl,z) & __hgt2_mask(cut2,d2);
         z = __hmax2_nan(__hgt2(cut2,d2)*zl,z);
         k = m&j2 | (~m)&k;
         xl = __lowhigh2highlow(xl);
         yl = __lowhigh2highlow(yl);
         zl = __lowhigh2highlow(zl);
         d2 = (xl-x)*(xl-x) + (yl-y)*(yl-y);
         m = __hgt2_mask(zl,z) & __hgt2_mask(cut2,d2);
         z = __hmax2_nan(__hgt2(cut2,d2)*zl,z);
         j2 = j*2+1 | (j*2)<<16;
         k = m&j2 | (~m)&k;
       }else {
        auto d2 = (x-lV.xc(j))*(x-lV.xc(j)) + (y-lV.yc(j))*(y-lV.yc(j));
        if (lV.zc(j)>z && d2<cut2) { k = j; z=lV.zc(j);}
//        bool doit = inV.zc(j)>z && d2<cut2;
//        k = doit ? j :k; 
//        z = doit ? inV.zc(j) : z;
       }
     }
     outV.x[start+i] = k;
   }
}

// blocks are independent events...
template<typename T, typename U, bool>
__global__ void benchit0(T inV,  U const outV, int const * __restrict__ ind) {

   bfN2(inV,outV,ind);
//   tile1D(inV,outV,ind);
}


template<typename T, typename U, bool>
__global__ void benchit1(T inV,  U const outV, int const * __restrict__ ind) {
//   bfN2(inV,outV,ind);
   tile1D(inV,outV,ind);
}


#ifndef NB
#define NB 188
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

  benchit0<T,U,G::TW><<<nB,nT,0,0>>>(a, b, ind);
  cudaDeviceSynchronize();
  if constexpr(std::is_same<half2, typename std::remove_cv<typename T::Type>::type>::value) return;
  benchit1<T,U,G::TW><<<nB,nT,0,0>>>(a, b, ind);
  cudaDeviceSynchronize();
}


#include<iostream>
int main() {

//   cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

   std::cout << toInd(-0.45f) << ' ' << toInd(0.45f)  << std::endl;


   int n = MX*NB;

   doBench<G<float,uint32_t,false>,SoA<float>,SoA<uint32_t>>("puf n2",n);
   doBench<G<half,uint32_t,false>,SoA<half>,SoA<uint32_t>>("puh n2",n);
   doBench<G<half2,uint32_t,false>,SoA<half2>,SoA<uint32_t>>("puh2 n2",n/2);
   doBench<G<float,uint32_t,true>,SoA<float>,SoA<uint32_t>>("2jf n2",n);
   doBench<G<half,uint32_t,true>,SoA<half>,SoA<uint32_t>>("2jh n2",n);
   doBench<G<half2,uint32_t,true>,SoA<half2>,SoA<uint32_t>>("2jh2 n2",n/2);
   return 0;
}
