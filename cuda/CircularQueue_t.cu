#define TEST_CIRCULAR_QUEUE
#include "CircularQueue.h"
#include<cassert>

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


#include<iostream>

int main() {

   using Q = CircularQueue<int,1024>;

   std::cout << "Qmaxs " << Q::maxCapacity << ' ' << Q::maxHead << std::endl;

   test0<<<1,1,0>>>();   
   cudaDeviceSynchronize();

   return 0;
}
