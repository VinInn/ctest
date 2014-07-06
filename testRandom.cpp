#include<random>
#include<iostream>


#include <thread>
#include <functional>
#include <algorithm>
#include<vector>
// convert gcc to c0xx
#define thread_local __thread

typedef std::thread Thread;
typedef std::vector<std::thread> ThreadGroup;
typedef std::mutex Mutex;
typedef std::unique_lock<std::mutex> Guard;
typedef std::condition_variable Condition;

#include<sstream>

long long threadId() {
  std::stringstream ss;
  ss << std::this_thread::get_id();
  long long id;
  ss >> id;
  return id;
}


// thread_local long long tid=threadId();

std::vector<double> v(50,0.);

volatile long start;
  void wait() {
    __sync_add_and_fetch(&start,-1);
    do{}while(start);
  }

void fun(int k) {
  wait();
  long long tid=threadId();
  std::mt19937 eng;
  eng.seed(tid);
  for (int i=k; i<k+5;i++)
    v[i]=double(eng()-eng.min())/double(eng.max()-eng.min()); 
}


int main() {

   std::mt19937 eng;
   std::mt19937 eng2;
   // std::ranlux_base_01 reng;
   std::ranlux_base_48 reng;
   std::uniform_int<int> ugen(1,50);
   std::uniform_real<double> rgen(-5.,5.);
   std::poisson_distribution<int> poiss(5.);
   std::normal_distribution<double> gauss(0.,1.);

   for (int i=0; i<5; i++) {
      std::cout << eng() << std::endl;
      std::cout << eng2() << std::endl;
      std::cout << double(eng()-eng.min())/double(eng.max()-eng.min()) << std::endl;
      std::cout << ugen(eng) << std::endl;
      std::cout << rgen(reng) << std::endl;
      std::cout << poiss(reng) << std::endl;
      std::cout << gauss(reng) << std::endl;
   }


 const int NUMTHREADS=10;
 start=NUMTHREADS;

  ThreadGroup threads;
  threads.reserve(NUMTHREADS);
  int k=0;
  for (int i=0; i<NUMTHREADS; ++i) {
    threads.push_back(Thread(fun,k));
    k+=5;
  }
  
  std::for_each(threads.begin(),threads.end(), 
		std::bind(&Thread::join,std::placeholders::_1));

  for (int i=0; i<50; i+=5) {
    for (int j=0; j<5; j++)
      std::cout << v[i+j]<<" ";
    std::cout << std::endl;
  }
  return 0;
}
