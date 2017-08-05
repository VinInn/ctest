// c++ -march=native -Ofast histFill.cpp -DHTMEM -pthread -mrtm
// c++ -march=native -Ofast histFill.cpp -fgnu-tm
#include <vector>
#include <cmath>
#include <limits>
#include <queue>
#include <mutex>
#include <thread>
#include <functional>
#include <algorithm>
#include <immintrin.h>
#include<atomic>
#include<random>
#include<functional>


// convert gcc to c0xx
// #define thread_local __thread;

typedef std::thread Thread;
typedef std::vector<std::thread> ThreadGroup;
typedef std::mutex Mutex;
typedef std::lock_guard<std::mutex> Lock;
// typedef std::unique_lock<std::mutex> Lock;
// typedef std::condition_variable Condition;

#ifdef NOTRANS
#define __transaction_atomic
#endif

std::atomic<long long> nfail(0);

#ifdef HTMEM

inline
unsigned int spin() {
  unsigned int status = _XABORT_EXPLICIT;
  for (int n_tries = 0; n_tries < 100; ++n_tries) 
  {
    status = _xbegin ();
    if (status == _XBEGIN_STARTED || !(status & _XABORT_RETRY))
      break;
  }
  return status;
}

#define __transaction_atomic unsigned int status=spin(); if (status != _XBEGIN_STARTED) {++nfail;} else
#else
void _xend (){}
#endif



struct Hist {

int tot() const { return std::accumulate(bins.begin(),bins.end(),0);}

void fill(int i) {
   __transaction_atomic {
       ++bins[i];
      _xend ();
   }  
}

std::array<int,100> bins = {{0}};

};


struct Ahist {

   Ahist(){}

int tot() const { return std::accumulate(bins.begin(),bins.end(),0);}


void fill(int i) {
       ++bins[i];
}

std::array<std::atomic<int>,100> bins;

};


#include<iostream>
// control cout....
Mutex outLock;

Hist hist;


bool stop=false;
bool working=true;
void stopper() {
  {
    Lock a(outLock);
    std::cout << "I'm the stopper" << std::endl;
  }

  //  char s;
  //std::cin >> s;
  while (working) {
    std::chrono::milliseconds dura( 2000 );
    std::this_thread::sleep_for( dura );
  }

  stop=true;
  {
    Lock a(outLock);
    std::cout << "stop sent " << std::endl;
  }

}


void act() {

  int me=0;


  std::default_random_engine generator;
  std::uniform_int_distribution<unsigned int> distribution(0,100);
  auto pos = std::bind ( distribution, generator );

  __transaction_atomic {
    // not really needed atomic will be enough
    static int nt=0;
    me = nt++;
    _xend();
  }  

  {
    Lock a(outLock);
    std::cout << "starting " <<  me << std::endl;
  }

  for (int i=0; i<1000000;++i)
    hist.fill(pos());

}

int main(){

 const int NUMTHREADS=10;
  ThreadGroup threads;
  threads.reserve(NUMTHREADS);
  for (int i=0; i<NUMTHREADS; ++i) {
    threads.push_back(Thread(act));
  }

  Thread s(stopper);
  s.detach();

  std::for_each(threads.begin(),threads.end(), 
	       std::bind(&Thread::join,std::placeholders::_1));
  

  std::cout << "total "<< hist.tot() << std::endl;

}
