#define TEST_CIRCULAR_QUEUE
#include "CircularQueue.h"
#include<cassert>
#include <cstdio>

__global__
void test0() {
  
  using Q = CircularQueue<int,1024>;

  __shared__ Q q;
  assert(q.constructEmpty(1024));
  assert(q.empty());
  assert(!q.full());
  assert(q.size()==0);
  assert(q.capacity()==1024);
  assert(-1==q.pop(-1));
  assert(q.empty());

  assert(q.push(1));
  assert(!q.empty());
  assert(!q.full());
  assert(q.size()==1);
  assert(q.capacity()==1024);
  assert(1==q.pop(-1));
  assert(q.empty());
  assert(!q.full());
  assert(q.size()==0);
  assert(q.capacity()==1024);
  assert(-1==q.pop(-1));
  assert(q.empty());



  assert(q.constructFull(1024));
  assert(!q.empty());
  assert(q.full());
  assert(q.size()==1024);
  assert(q.capacity()==1024);
  assert(!q.push(1));
  assert(q.full());
  q.pop(-1);
  assert(!q.empty());
  assert(!q.full());
  assert(q.size()==1023);
  assert(q.push(1));
  assert(!q.empty());
  assert(q.full());
  assert(q.size()==1024);
  assert(q.capacity()==1024);
  assert(!q.push(1));
  assert(q.full());

}

__global__
void testSimple() {

  assert(blockDim.x==1024);
  using Q = CircularQueue<int,1024>;

  __shared__ Q q;
  assert(q.constructFull(1024));

  int i = threadIdx.x;
  q.data()[i]=i;
  __syncthreads();
  if (0==i) printf("h,t %d %d\n",q.head(),q.tail());
  __syncthreads();

  auto k = q.pop(-1);
  assert(k>=0);assert(k<1024);
  __syncthreads();
  assert(q.empty());
  assert(-1==q.pop(-1));
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

  k = q.pop(-1);
  if (k%2) assert(q.push(k));
  __syncthreads();
  assert(!q.full());
  assert(!q.empty());
  if (0==i) printf("h,t s %d %d %d\n",q.head(),q.tail(),q.size());
  __syncthreads();
  k = q.pop(-1);
  if (k>=0) assert(k%2);
  __syncthreads();
  assert(q.empty());
  if (0==i) printf("h,t s %d %d %d\n",q.head(),q.tail(),q.size());
  __syncthreads();

  assert(q.push(i));
  __syncthreads();
  assert(q.full());
  __syncthreads();

  k = q.pop(-1);
  assert(k>=0); assert(k<1024);
  if (k%2) q.unsafePush(k);
  __syncthreads();
  assert(!q.full());
  assert(!q.empty());
  if (0==i) printf("h,t s %d %d %d\n",q.head(),q.tail(),q.size());
  __syncthreads();
  k = q.pop(-1);
  if (k>0) assert(k%2);
  __syncthreads();
  assert(q.empty());
  if (0==i) printf("h,t s %d %d %d\n",q.head(),q.tail(),q.size());
  __syncthreads();



}


#include<iostream>

int main() {

   using Q = CircularQueue<int,1024>;

   std::cout << "Qmaxs " << Q::maxCapacity << ' ' << Q::maxHead << std::endl;

   test0<<<1,1,0>>>();   
   cudaDeviceSynchronize();
   testSimple<<<1,1024,0>>>();
   cudaDeviceSynchronize();

   return 0;
}
