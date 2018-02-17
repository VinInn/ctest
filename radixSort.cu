#include<cstdint>
#include<cassert>

__global__  
void radixSort(int16_t * v, uint32_t * ind, uint32_t size) {
     
  constexpr int d = 8, w = 16;
  constexpr int sb = 1<<d;

  constexpr int MaxSize = 256*32;
  __shared__ uint32_t ind2[MaxSize];
  __shared__ uint32_t c[sb];
  __shared__ uint32_t firstNeg;    

  assert(size<=MaxSize);  // for multiple blocks this is not correct
  assert(blockDim.x==sb);  

  assert(blockIdx.x==0);
  // int first = blockDim.x * blockIdx.x + threadIdx.x;

  firstNeg=0;

  auto a = v; // later add offset
  auto j = ind; // later add offset
  auto k = ind2;

  int32_t first = threadIdx.x;
  for (auto i=first; i<size; i+=blockDim.x)  j[i]=i;
  __syncthreads();


  for (int p = 0; p < w/d; ++p) {
    c[threadIdx.x]=0;
    __syncthreads();

    // fill bins
    for (auto i=first; i<size; i+=blockDim.x) 
      atomicAdd(&c[(a[j[i]] >> d*p)&(sb-1)],1);
    __syncthreads();

    // prefix scan to be optimized...
    if (threadIdx.x==0)
      for (int i = 1; i < sb; ++i) c[i] += c[i-1];
    __syncthreads();

    // broadcast
    if (threadIdx.x==0)
    for (int i=size-first-1; i>=0; i--) { // =blockDim.x) {
      auto ik = atomicSub(&c[(a[j[i]] >> d*p)&(sb-1)],1);
      k[ik-1] = j[i];
    }
    __syncthreads();

    // swap (local, ok)
    auto t=j;j=k;k=t;
  }

  // w/d is even so ind is correct
  assert(j==ind);
  __syncthreads();

  

  // now move negative first...
  // find first negative
  for (auto i=first; i<size-1; i+=blockDim.x) {
    // if ( (int(a[ind[i]])*int(a[ind[i+1]])) <0 ) firstNeg=i+1;
   if ( (a[ind[i]]^a[ind[i+1]]) < 0 ) firstNeg=i+1; 
  }
  
  __syncthreads();
  assert(firstNeg>0);

  auto ii=first;
  for (auto i=firstNeg+threadIdx.x; i<size; i+=blockDim.x)  { ind2[ii] = ind[i]; ii+=blockDim.x; }
  __syncthreads();
  ii= size-firstNeg +threadIdx.x;
  assert(ii>=0);
  for (auto i=first;i<firstNeg;i+=blockDim.x)  { ind2[ii] = ind[i]; ii+=blockDim.x; }
  __syncthreads();
  for (auto i=first; i<size; i+=blockDim.x) ind[i]=ind2[i];

  
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

 std::cout << v[ind[0]] << ' ' << v[ind[1]] << ' ' << v[ind[2]] << std::endl;
   std::cout << v[ind[3]] << ' ' << v[ind[10]] << ' ' << v[ind[N-1000]] << std::endl;
  std::cout << v[ind[N/2-1]] << ' ' << v[ind[N/2]] << ' ' << v[ind[N/2+1]] << std::endl;
 for (int i = 1; i < N; i++) {
    if (v[ind[i]]<v[ind[i-1]])
      std::cout << "not ordered at " << ind[i] << " : "
		<< v[ind[i]] <<' '<< v[ind[i-1]] << std::endl;
 }
  return 0;
}
