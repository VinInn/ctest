#define TEST_CIRCULAR_QUEUE
#include "CircularQueue.h"
#include<cassert>
#include <cstdio>


#define GPU_DEBUG
#include "radixSort.h"


__global__
void test0() {
  
  using Q = CircularQueue<int,1024>;

  __shared__ Q q;
  assert(q.construct(1024));
  for (int j=0;j<1024*2;++j) q.data()[j]=-1;
  printf("empty ht %d %d\n",*q.headTail());
  assert(q.empty());
  assert(!q.full());
  assert(q.size()==0);
  assert(q.capacity()==1024);
  assert(q.mask()==1023);
  assert(q.head()==1024);
  assert(q.tail()==0);
  assert(-1==q.pop());
  assert(q.empty());

  printf("h,t s %d %d %d\n",q.head(),q.tail(),q.size());

  assert(q.push(1));
  printf("one ht %d %d\n",*q.headTail());

  assert(!q.empty());
  assert(!q.full());
  assert(q.size()==1);
  assert(q.capacity()==1024);
  assert(1==q.pop());
  assert(-1==q.pop());
  // assert(q.empty());

  printf("h,t s %d %d %d\n",q.head(),q.tail(),q.size());
  assert(-1==q.pop());
  printf("h,t s %d %d %d\n",q.head(),q.tail(),q.size());

  assert(q.construct(1024));
  for (int j=0;j<1023;++j) q.data()[j]=j;
  for (int j=1023;j<1024*2;++j) q.data()[j]=-1;
  printf("full ht %d %d\n",*q.headTail());
  assert(q.capacity()==1024);
  assert(q.head()==1024);
  assert(q.tail()==0);
  assert(q.push(99991));
  assert(0==q.pop());
  assert(1==q.pop());
  assert(q.size()==1);
  assert(q.push(99992));
  assert(!q.empty());
  assert(!q.full());
  assert(q.size()==2);
  assert(q.capacity()==1024);
  assert(q.push(99993));
  assert(q.size()==3);


  printf("ht %d %d\n",*q.headTail());

  printf("h,t s %d %d %d\n",q.head(),q.tail(),q.size());



}


__global__
void testSimple() {

  assert(blockDim.x==1024);
  using Q = CircularQueue<int,1024>;

  __shared__ Q q;
  assert(q.construct(1024));

  int i = threadIdx.x;
  q.data()[i]=i;
  q.data()[1023] = -1;
  q.data()[i+1024]=-1;
  __syncthreads();
  if (0==i) printf("h,t %d %d\n",q.head(),q.tail());
  __syncthreads();

  if(i!=0) {
     auto k = q.pop();
     assert(k>=0);assert(k<1024);
  }
  __syncthreads();
  assert(q.empty());
  assert(-1==q.pop());
  __syncthreads();
  if (0==i) printf("h,t %d %d\n",q.head(),q.tail());
  __syncthreads();
  assert(q.push(i));
  __syncthreads();
  if (0==i) printf("h,t %d %d\n",q.head(),q.tail());
  __syncthreads();
  assert(q.full());
  assert(!q.push(i));
  __syncthreads();
  if (0==i) printf("h,t %d %d\n",q.head(),q.tail());
  __syncthreads();

  if(i!=0) {
    auto k = q.pop();
    assert(k>=0); assert(k<1024);
    if (k%2) q.push(k);
  }
  __syncthreads();
  assert(!q.full());
  assert(!q.empty());
  if (0==i) printf("h,t s %d %d %d\n",q.head(),q.tail(),q.size());
  __syncthreads();
  auto k = q.pop();
  if (k>0) assert(k%2);
  __syncthreads();
  assert(q.empty());
  if (0==i) {
    printf("h,t s %d %d %d\n",q.head(),q.tail(),q.size());
    printf("ht %d %d\n",*q.headTail());
  }
  __syncthreads();

  int jj=0;
  while(__syncthreads_or(jj<16)) {
    if (jj>=7 && 0==i) {
      printf("jj %d: h,t s %d %d %d ",jj,q.head(),q.tail(),q.size()); 
      printf("ht %d %d\n",*q.headTail());
    }
    k = q.pop();
    if (k>=0 && k%2) q.push(k);
    q.push(jj);
    ++jj;
  }

  // just a mess
  for (int kk=0; kk<32; ++kk) {
    k = q.pop();
    if (k>=0 && k%2) q.push(k);
    if (k>=0 && k%3) q.push(kk);
  }
  __syncthreads();
  if (0==i) {
    printf("h,t s %d %d %d\n",q.head(),q.tail(),q.size());
    printf("ht %d %d\n",*q.headTail());

    q.construct(1024);
  }

  q.data()[i]=i;
  q.data()[1023]=-1;
  q.data()[i+1024]=-1;

  __shared__ int v[1024];
  __shared__ int vc;
  v[i]=-2;
  vc=0;
  __syncthreads();

  assert(q.full());
  assert(0==q.data()[0]); 
  assert(1023==q.data()[1023]);
  __syncthreads();

  for (int kk=0; kk<32; ++kk) {
    k = q.pop();
    if (k<0) continue;
    if ((k+kk)%2) q.push(k);
    else {
      auto j = atomicAdd(&vc,1);
      assert(j<1024);
      v[j]=k;
    }
  }

  __syncthreads();
  if (0==i) printf("filled %d\n",vc);
  if (0==i) {
    printf("h,t s %d %d %d\n",q.head(),q.tail(),q.size());
  }
  __shared__ uint16_t sortInd[1024];
  __shared__ uint16_t sws[1024];
  radixSort(v,sortInd,sws,vc);
  __syncthreads();
  if (0==i) printf("sorted %d %d\n",v[sortInd[0]],v[sortInd[vc-1]]);
  assert(v[i]!=-1);
  if (i<vc) { assert(v[i]>=0); assert(v[i]<1024); }
  if(i>0 && i<vc) if (!(v[sortInd[i-1]]<v[sortInd[i]])) 
         printf("mess %d %d %d %d\n",sortInd[i-1],sortInd[i],v[sortInd[i-1]],v[sortInd[i]]);
 
}





#include<iostream>

int main() {

   using Q = CircularQueue<int,1024>;

   std::cout << "Qmaxs " << Q::maxCapacity << ' ' << Q::maxHead << std::endl;

   Q q; q.construct(1024);
   std::cout << "seg " << q.segment(0) << ' ' << q.segment(1023) << ' ' << q.segment(1024) << ' ' << q.segment(1024+1023) << std::endl;

   std::cout << "\ntest0" << std::endl;
   test0<<<1,1,0>>>();   
   cudaDeviceSynchronize();
   std::cout << "\ntestSimple" << std::endl;
   testSimple<<<1,1024,0>>>();
   cudaDeviceSynchronize();

   return 0;
}
