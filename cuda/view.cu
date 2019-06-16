#include <cuda_runtime.h>
// Type your code here, or load an example.
struct View {
  float *mx, *my, *mz;
  __device__ __forceinline__ float& x(int i) { return mx[i]; }
  __device__ __forceinline__ float  x(int i) const { return __ldg(mx + i); }
  __device__ __forceinline__ float& y(int i) { return my[i]; }
  __device__ __forceinline__ float  y(int i) const { return __ldg(my + i); }
  __device__ __forceinline__ float& z(int i) { return mz[i]; }
  __device__ __forceinline__ float  z(int i) const { return __ldg(mz + i); }
};

__global__ void cross(View const * pvi, View * pvo, int n) {
    auto const & vi = *pvi;
    auto & vo = *pvo;
    int tid = blockIdx.x;
    if (tid >= n) return; 
    vo.x(tid) = vi.y(tid)*vi.z(tid);
    vo.y(tid) = vi.x(tid)*vi.z(tid);
    vo.z(tid) = vi.x(tid)*vi.y(tid);
}

__global__ void crossO(View const * pvi, View * pvo, int n) {
    auto const & vi = *pvi;
    auto & vo = *pvo;
    int tid = blockIdx.x;
    if (tid >= n) return;
    auto x =  vi.x(tid);
    auto y =  vi.y(tid);
    auto z =  vi.z(tid);
    vo.x(tid) = y*z;
    vo.y(tid) = x*z;
    vo.z(tid) = x*y;
}


__global__ void cross(float const * __restrict__ xi, float const * __restrict__ yi, float const * __restrict__ zi,
                    float *xo, float *yo, float *zo, int n){
    int tid = blockIdx.x;
    if (tid >= n) return; 
    xo[tid] = yi[tid]*zi[tid]; 
    yo[tid] = xi[tid]*zi[tid]; 
    zo[tid] = xi[tid]*yi[tid]; 
}

