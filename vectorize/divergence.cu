// nvcc -O3 -std=c++14 --expt-relaxed-constexpr -gencode arch=compute_70,code=sm_70 divergence.cu
#include<cmath>
#include<iostream>
#include<memory>

__global__
void set(double * v, int n, int flag, double * __restrict__ res) {
  auto first = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i=first; i<n; i+=gridDim.x*blockDim.x) {
    v[i]=0;
    if (1==flag && 0==(i%512)) v[i]=M_PI;
    if (2==flag && 0==(i%128)) v[i]=M_PI;
    if (3==flag && 0==(i%4)) v[i]=M_PI;
    if (4==flag) v[i]=M_PI;
  }
    *res=0;
}

__global__
void compute(double const * __restrict__ v, int n, double * __restrict__ res) {
  auto first = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i=first; i<n; i+=gridDim.x*blockDim.x)
    //  *res += (v[i]>0) ? 1./std::sqrt(v[i]) : 0;
    // if (v[i]>0) atomicAdd(res,1./std::sqrt(v[i]));
    if (v[i]>0) res[i] = 1./std::sqrt(v[i]);

}




int main(int argc, char**) {

  std::cout << "flag " << argc-1 << std::endl;

  int size=16*1024;
  double * v;
  cudaMalloc(&v,(size+1)*sizeof(double));

  double * res; // = v+size;
  cudaMalloc(&res,(size+1)*sizeof(double));
  set<<<256,48>>> (v,size,argc-1,res);  

  for (int i=0; i<500000; ++i) {
    compute<<<4,64>>>(v,size,res);
  }

  cudaFree(v);
  return 0;

}
