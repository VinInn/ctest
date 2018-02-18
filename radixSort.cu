#include<cstdint>
#include<cassert>
#include<cstdio>

template<typename T>
__device__  
void radixSort(T * a, uint16_t * ind, uint32_t size) {
    
  constexpr int d = 8, w = 8*sizeof(T);
  constexpr int sb = 1<<d;

  constexpr int MaxSize = 256*32;
  __shared__ uint16_t ind2[MaxSize];
  __shared__ int32_t c[sb], ct[sb], cu[sb];
  __shared__ uint32_t firstNeg;    

  assert(size<=MaxSize); 
  assert(blockDim.x==sb);  

  // bool debug = false; // threadIdx.x==0 && blockIdx.x==5;

  firstNeg=0;

  auto j = ind;
  auto k = ind2;

  int32_t first = threadIdx.x;
  for (auto i=first; i<size; i+=blockDim.x)  j[i]=i;
  __syncthreads();


  for (int p = 0; p < w/d; ++p) {
    c[threadIdx.x]=0;
    __syncthreads();

    // fill bins
    for (auto i=first; i<size; i+=blockDim.x) {
      auto bin = (a[j[i]] >> d*p)&(sb-1);
      atomicAdd(&c[bin],1);
    }
    __syncthreads();

    // prefix scan "optimized"???...
    auto x = c[threadIdx.x];
    auto laneId = threadIdx.x & 0x1f;
    #pragma unroll
    for( int offset = 1 ; offset < 32 ; offset <<= 1 ) {
      auto y = __shfl_up_sync(0xffffffff,x, offset);
      if(laneId >= offset) x += y;
    }
    ct[threadIdx.x] = x;
    __syncthreads();
    auto ss = (threadIdx.x/32)*32 -1;
    c[threadIdx.x] = ct[threadIdx.x];
    for(int i=ss; i>0; i-=32) c[threadIdx.x] +=ct[i]; 

    /* prefix scan for the nulls
    if (threadIdx.x==0)
      for (int i = 1; i < sb; ++i) c[i] += c[i-1];
    */
    __syncthreads();


    for (int i=size-first-1; i>=0; i-=blockDim.x) {
       cu[threadIdx.x]=-1;
       auto bin = (a[j[i]] >> d*p)&(sb-1);
       ct[threadIdx.x]=bin;
       atomicMax(&cu[bin],int(i));
       __syncthreads();
       if (i==cu[bin])  // ensure to keep them in order
         for (int ii=threadIdx.x; ii<blockDim.x; ++ii) if (ct[ii]==bin) {auto oi = ii-threadIdx.x; k[--c[bin]] = j[i-oi]; }
       __syncthreads();
     }    
 
    /*  
    // broadcast for the nulls
    if (threadIdx.x==0)
    for (int i=size-first-1; i>=0; i--) { // =blockDim.x) {
      auto ik = atomicSub(&c[(a[j[i]] >> d*p)&(sb-1)],1);
      k[ik-1] = j[i];
    }
    */

    __syncthreads();
    assert(c[0]==0);


    // swap (local, ok)
    auto t=j;j=k;k=t;
  }

  // w/d is even so ind is correct
  assert(j==ind);
  __syncthreads();

  

  // now move negative first...
  // find first negative  (for float ^ will not work...)
  for (auto i=first; i<size-1; i+=blockDim.x) {
    // if ( (int(a[ind[i]])*int(a[ind[i+1]])) <0 ) firstNeg=i+1;
   if ( (a[ind[i]]^a[ind[i+1]]) < 0 ) firstNeg=i+1; 
  }
  
  __syncthreads();
  // assert(firstNeg>0); not necessary true if all positive !

  auto ii=first;
  for (auto i=firstNeg+threadIdx.x; i<size; i+=blockDim.x)  { ind2[ii] = ind[i]; ii+=blockDim.x; }
  __syncthreads();
  ii= size-firstNeg +threadIdx.x;
  assert(ii>=0);
  for (auto i=first;i<firstNeg;i+=blockDim.x)  { ind2[ii] = ind[i]; ii+=blockDim.x; }
  __syncthreads();
  for (auto i=first; i<size; i+=blockDim.x) ind[i]=ind2[i];

  
}


template<typename T>
__global__
void radixSortMulti(T * v, uint16_t * index, uint32_t * offsets) {

  auto a = v+offsets[blockIdx.x];
  auto ind = index+offsets[blockIdx.x];;
  auto size = offsets[blockIdx.x+1]-offsets[blockIdx.x];

  radixSort(a,ind,size);

}

#include "cuda/api_wrappers.h"

#include <iomanip>
#include <memory>
#include <algorithm>
#include <chrono>
#include<random>


#include<cassert>
#include<iostream>
#include<limits>


template<typename T>
void go() {

std::mt19937 eng;
// std::mt19937 eng2;
std::uniform_int_distribution<T> rgen(std::numeric_limits<T>::min(),std::numeric_limits<T>::max());


  auto start = std::chrono::high_resolution_clock::now();
  auto delta = start - start;

  if (cuda::device::count() == 0) {
	std::cerr << "No CUDA devices on this system" << "\n";
	exit(EXIT_FAILURE);
  }

  auto current_device = cuda::device::current::get(); 

  constexpr int blocks=10;
  constexpr int blockSize = 256*32;
  constexpr int N=blockSize*blocks;
  T v[N];
  uint16_t ind[N];

  std::cout << "Will sort " << N << " 'ints' of size " << sizeof(T) << std::endl;


  for (int i=0; i<50; ++i) {

    if (i==49) { 
        for (long long j = 0; j < N; j++) v[j]=0;
    } else if (i>30) {
    for (long long j = 0; j < N; j++) v[j]=rgen(eng);
    } else {
      long long imax = (i<15) ? std::numeric_limits<T>::max() +1LL : 255;
      for (long long j = 0; j < N; j++) {
        v[j]=(j%imax); if(j%2 && i%2) v[j]=-v[j];
      }
    }

  uint32_t offsets[blocks+1];
  offsets[0]=0;
  for (int j=1; j<blocks+1; ++j) offsets[j] = offsets[j-1]+blockSize;


  std::random_shuffle(v,v+N);
  auto v_d = cuda::memory::device::make_unique<T[]>(current_device, N);
  auto ind_d = cuda::memory::device::make_unique<uint16_t[]>(current_device, N);
  auto off_d = cuda::memory::device::make_unique<uint32_t[]>(current_device, blocks+1);

  cuda::memory::copy(v_d.get(), v, N*sizeof(T));
  cuda::memory::copy(off_d.get(), offsets, 4*(blocks+1));


   int threadsPerBlock =256;
   int blocksPerGrid = blocks;
   delta -= (std::chrono::high_resolution_clock::now()-start);
   cuda::launch(
                radixSortMulti<T>,
                { blocksPerGrid, threadsPerBlock },
                v_d.get(),ind_d.get(),off_d.get()
        );


//  cuda::memory::copy(v, v_d.get(), 2*N);
   cuda::memory::copy(ind, ind_d.get(), 2*N);

   delta += (std::chrono::high_resolution_clock::now()-start);

  if (32==i) {
    std::cout << v[ind[0]] << ' ' << v[ind[1]] << ' ' << v[ind[2]] << std::endl;
    std::cout << v[ind[3]] << ' ' << v[ind[10]] << ' ' << v[ind[blockSize-1000]] << std::endl;
    std::cout << v[ind[blockSize/2-1]] << ' ' << v[ind[blockSize/2]] << ' ' << v[ind[blockSize/2+1]] << std::endl;
  }
  for (int ib=0; ib<blocks; ++ib)
  for (int i = offsets[ib]+1; i < offsets[ib+1]; i++) {
      auto a = v+offsets[ib];
   // assert(!(a[ind[i]]<a[ind[i-1]]));
     if (a[ind[i]]<a[ind[i-1]])
      std::cout << ib << " not ordered at " << ind[i] << " : "
  		<< a[ind[i]] <<' '<< a[ind[i-1]] << std::endl;
  }
 }  // 50 times
     std::cout <<"cuda computation took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()/50.
              << " ms" << std::endl;
}


int main() {

  go<int16_t>();
  go<int32_t>();
  return 0;
}
