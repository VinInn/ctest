#include<vector>
#include <thread>
#include <mutex>
#include <functional>
#include <algorithm>
#include<cmath>
#include<iostream>

#include<omp.h>

#include "CircularBuffer.h"


typedef std::mutex Mutex;
// typedef std::lock_guard<std::mutex> Lock;
typedef std::unique_lock<std::mutex> Lock;
typedef std::unique_lock<std::mutex> Guard;
//typedef std::condition_variable Condition;


namespace global {
  // control cout....
  Mutex coutLock;
}

int pushing[100]={0,};

struct Pusher {

  explicit Pusher(CircularBuffer<int>& ib) : b(ib){}

  bool operator()() {
    //{ Guard g(global::coutLock); 
    // std::cout << "pushing "<< omp_get_thread_num() << " : " << k << " " << b.size() << std::endl; 
    // }
    if (k>=imax) return false;
    auto lm = std::min(imax,k+1000);
    for (; k<lm; ++k) 
      if (!b.push(k%100,false)) break;
    //if (!b.push(k%100,true)) break; // this is stupid!!! 
    // { Guard g(global::coutLock); 
    //   std::cout << "pushed "<< omp_get_thread_num() << " : " << k << " " << b.size() << std::endl; 
    // }

    if (k>=imax) { b.drain(); return false;}
    return true;
  }

  int k=0;
  int imax=100000;
  CircularBuffer<int>& b;
};

struct Worker {

  Worker(CircularBuffer<int>& ib, Pusher & ip) : b(ib), p(ip), hist(100){}
  
  void operator()() {
    int i;
    while (true) {
      // first try to push
      if ( (!b.draining()) && b.halfEmpty() && b.tryLock()) {
	p();
	b.releaseLock();
	++pushing[omp_get_thread_num()];
      }
      int k=100;
      while( (b.draining() || (--k))  && b.pop(i, false)) {
	++hist[i];
      }
      if (b.draining()) break;
    }
    // needed???
    while( b.pop(i)) {
      ++hist[i];
    }

  }
  
  CircularBuffer<int>& b;
  Pusher & p;
  std::vector<int> hist;

};



int main() {
  

  CircularBuffer<int> buff(30);
  Pusher p(buff);

  auto NUMTHREADS = omp_get_max_threads();

  std::vector<Worker> workers(NUMTHREADS, Worker(buff,p));

#pragma omp parallel
  {

    auto me = omp_get_thread_num();

    /*
    if (me==0) {
      for (int i=0; i<100000; i++) 
	buff.push(i%100);
	buff.drain();
    }
    */

    workers[me]();

  }
 
  std::cout << "pushers: ";
  for (int i=0; i<NUMTHREADS; ++i) std::cout << pushing[i] << ", ";
  std::cout << std::endl;std::cout << std::endl;
    
  std::vector<int> hist(100);
  for (int i=0; i!=NUMTHREADS;++i) {
    { Guard g(global::coutLock); std::cout << "thread "<< i << " : "; }
    for (int j=0; j!=100;++j) 
      hist[j]+= workers[i].hist[j];
    {
      Guard g(global::coutLock);
      for (int j=0; j!=100;++j) 
	std::cout << workers[i].hist[j] << " ,";    
      std::cout << std::endl;
    }
  }
  
  {
    Guard g(global::coutLock);
    std::cout << "\nTotal " << std::endl;
    for (int j=0; j!=100;++j) 
      std::cout << hist[j] << ", ";
    std::cout << std::endl;
  }
  
  return 0;

}
