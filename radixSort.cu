#include<cstdint>
#include<cassert>

__global__  
void radixSort(int16_t * a, uint32_t * ind, uint32_t size) {
     
  constexpr int d = 8, w = 16;
  constexpr int sb = 1<<d;

  constexpr int MaxSize = 256*32;
  __shared__ uint32_t ind2[MaxSize];
  __shared__ uint32_t c[sb];
    
  assert(size<=MaxSize);  // for multiple blocks this is not correct
  assert(blockDim.x==sb);  

  int first = blockDim.x * blockIdx.x + threadIdx.x;
  
  for (auto i=first; i<size; i+=blockDim.x)  ind[i]=i;
  __syncthreads();

  auto j = ind;
  auto k = ind2;
  for (int p = 0; p < w/d; p++) {
    c[threadIdx.x]=0;
  __syncthreads();

    // fill bins
    for (auto i=first; i<size; i+=blockDim.x) 
      atomicAdd(&c[(a[j[i]] >> d*p)&(sb-1)],1);

   __syncthreads();
   // prefix scan to be optimized...
   if (first==0)
   for (int j = 1; j < sb; j++) c[j] += c[j-1];
   __syncthreads();

   // broadcast
   for (auto i=first; i<size; i+=blockDim.x) {
     auto ik = atomicSub(&c[(a[j[i]] >> d*p)&(sb-1)],1);
     k[ik-1] = j[i];
   }
   __syncthreads();

   // swap (local, ok)
   auto t=j;j=k;k=t;
  }

  assert(j==ind);
  // not needed if w/d is even
  // if (j!=ind)  for (auto i=first; i<size; i+=blockDim.x) ind[i]=j[i];

}


#include "cuda/api_wrappers.h"

#include <iomanip>
#include <memory>
#include <algorithm>
#include <chrono>
#include<random>


#include<cassert>
#include<iostream>

int main() {

  auto start = std::chrono::high_resolution_clock::now();
  auto delta = start - start;

	if (cuda::device::count() == 0) {
		std::cerr << "No CUDA devices on this system" << "\n";
		exit(EXIT_FAILURE);
	}

        auto current_device = cuda::device::current::get(); 


  constexpr int N=256*32;
  int16_t v[N];
  uint32_t ind[N];

  std::cout << "Will sort " << N << " shorts" << std::endl;

  for (int i = 0; i < N; i++) {
    v[i]=i%32768; if(i%2) v[i]=-v[i];
  }

  auto v_d = cuda::memory::device::make_unique<int16_t[]>(current_device, N);
  auto ind_d = cuda::memory::device::make_unique<uint32_t[]>(current_device, N);
  cuda::memory::copy(v_d.get(), v, 2*N);

   int threadsPerBlock =256;
   int blocksPerGrid = 1;
   delta -= (std::chrono::high_resolution_clock::now()-start);
   cuda::launch(
                radixSort,
                { blocksPerGrid, threadsPerBlock },
                v_d.get(),ind_d.get(),N
        );


   delta += (std::chrono::high_resolution_clock::now()-start);
   std::cout <<"cuda computation took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()
              << " ms" << std::endl;

//  cuda::memory::copy(v, v_d.get(), 2*N);
  cuda::memory::copy(ind, ind_d.get(), 4*N);

  std::cout << v[ind[10]] << ' ' << v[ind[N-1000]] << std::endl;
  std::cout << v[ind[N/2-1]] << ' ' << v[ind[N/2]] << ' ' << v[ind[N/2+1]] << std::endl;
  for (int i = 1; i < N; i++) {
    assert(v[ind[i]]>=v[ind[i-1]]);
  }
  return 0;
}
