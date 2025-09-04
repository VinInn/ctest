// /usr/local/cuda/bin/nvcc -gencode arch=compute_75,code=sm_75 -O3 clock.cu -DCLOCK -DFLOAT=float
#include "cstdint"

using Float = FLOAT;

// Type your code here, or load an example.
__global__ void square(Float* array,  int64_t * tt, int64_t * tg, int n) {
     __shared__ uint64_t gstart, gend;
     uint64_t start, end;
     int tid = blockDim.x * blockIdx.x + threadIdx.x;

     auto k = array[tid];

     if (tid==0) {
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
//        array[tid] = array[tid] * array[tid];
//        array[tid] = array[tid] * array[tid] +k;
      k = k*k;
      k = k*k+k;
    }
    // Record end time 
#ifdef CLOCK
       tt[tid] = clock64() -s;
#else
   asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end));
   tt[tid] = end - start;
#endif

    __syncthreads();
    if (tid==0) {
 #ifdef CLOCK
      *tg = clock64() -gstart;
#else
     asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gend));
     *tg = gend - gstart;
#endif
   }
   array[tid] = k;
}

#include<iostream>

int main(int argc, char** argv) {

  int n = 32;
  Float * a;
  int64_t * tt;
  int64_t * tg;


  cudaMallocManaged(&a, n*sizeof(Float));
  cudaMallocManaged(&tt, n*sizeof(int64_t));
  cudaMallocManaged(&tg, sizeof(int64_t));


  for (int i=0; i<n; ++i) a[i]=i;

  for (int i=0; i<n; ++i) tt[i]=0;
  *tg=0;
  square<<<1,32,0,0>>>(a,tt,tg,n);
  cudaDeviceSynchronize();
  for (int i=0; i<n; ++i) std::cout << a[i] <<  ' ';
  std::cout << std::endl;


  for (int i=0; i<n; ++i) std::cout << tt[i] <<  ' ';
  std::cout << '\n' << *tg << std::endl;
}
