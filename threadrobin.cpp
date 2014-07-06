#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <queue>
#include <thread>
#include <functional>
#include <algorithm>

// convert gcc to c0xx
#define thread_local __thread;

typedef std::thread Thread;
typedef std::vector<std::thread> ThreadGroup;
typedef std::mutex Mutex;
// typedef std::lock_guard<std::mutex> Lock;
typedef std::unique_lock<std::mutex> Lock;
typedef std::condition_variable Condition;

namespace {

  typedef std::vector<unsigned char> Buffer; 
  std::queue<Buffer > theQueue;
  Mutex lock;
  Condition doit;

  // control cout....
  Mutex outLock;

  bool stop=false;

  void longHeavyInit() {
     sleep(3);
     Lock(outLock);
      std::cout << " Initialize common data " << std::this_thread::get_id() << std::endl;
  }

  void readFromSocket() {
    static std::once_flag initialized;
    std::call_once(initialized, longHeavyInit);
    static __thread int n=0;

    {
      Lock(outLock);
      std::cout << " starting thread " << std::this_thread::get_id() << std::endl;
    }

    while(true) {
      unsigned int s = 10*(double(rand())/
			   double(std::numeric_limits<unsigned int>::max()));
      // fake wait;
      sleep(s);
      // fake read
      Buffer buffer(10); buffer[0] = s;
      { 
	Lock gl(lock);
        if (stop) break;
	// avoid copy...
	theQueue.push(Buffer());
	theQueue.back().swap(buffer);
	doit.notify_all();	
      }
      ++n;
    }
    
   {
      Lock(outLock);
      std::cout << "terminating thread " << std::this_thread::get_id() 
                << " read " << n << std::endl;
   }

  }

}

void stopper() {
  char s;
  std::cin >> s;
  stop=true;
}

int main (int argc, char **argv)
{
  Thread s(stopper);
  s.detach();
  const int NUMTHREADS=10;
//  Thread t();	
//  ThreadGroup threads(NUMTHREADS, std::ref(t));
//  std::for_each(threads.begin(),threads.end(),
//               std::bind(&Thread::swap,std::placeholders::_1, Thread(readFromSocket))
//               );
  ThreadGroup threads;
  threads.reserve(NUMTHREADS);
  for (int i=0; i<NUMTHREADS; ++i) {
    //Thread t(readFromSocket);
    threads.push_back(Thread(readFromSocket));
    //threads.back().swap(t);
  }
  
  Buffer buffer;
  while(true) {
    {
      Lock gl(lock);
      if (theQueue.empty()) doit.wait(gl);
      theQueue.front().swap(buffer);
      theQueue.pop();
    }
    // operate on buffer
    std::cerr << int(buffer[0]) << ",";
    
  }

  std::for_each(threads.begin(),threads.end(), 
	       std::bind(&Thread::join,std::placeholders::_1));
  
  return 0;
}
