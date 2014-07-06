#include<pmmintrin.h>
#include<iostream>

#include<vector>

#include <thread>
#include <functional>
#include <algorithm>
#include<cmath>

#include<iostream>

// convert gcc to c0xx
#define thread_local __thread;

typedef std::thread Thread;
typedef std::vector<std::thread> ThreadGroup;
typedef std::mutex Mutex;
typedef std::unique_lock<std::mutex> Guard;
typedef std::condition_variable Condition;


long long * volatile lock=0;

void fun() {

  _mm_monitor(lock,0,0);
  _mm_mwait(0,0);


  std::cout << "hi " << lock[0] << std::endl;

}


int main() {

  lock = new long long [10];

  Thread p1(fun);

  p1.join();

  std::cout << "main " << lock[0] << std::endl;

  
  return 0;
 
}



