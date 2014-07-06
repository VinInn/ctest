#include<omp.h>
#include <mutex>
typedef std::mutex Mutex;
// typedef std::lock_guard<std::mutex> Lock;
typedef std::unique_lock<std::mutex> Lock;

namespace global {
  // control cout....
  Mutex coutLock;
}

#include <iostream>
#include <atomic>
#include <thread>


template<typename T>
inline void spinlock(std::atomic<T> const & lock, T val) {
  while (lock.load(std::memory_order_acquire)!=val){}
}



template<typename T>
inline void spinlockSleep(std::atomic<T> const & lock, T val) {
  while (lock.load(std::memory_order_acquire)!=val){nanosleep(0,0);}
}

// syncronize all threads in a parallel section (for testing purposes)
void sync(std::atomic<int> & all) {
  auto sum = omp_get_num_threads(); sum = sum*(sum+1)/2;
  all.fetch_add(omp_get_thread_num()+1,std::memory_order_acq_rel);
  spinlock(all,sum);

}


int main() {

  for (int i=0; i<10; i++) {
    int a=0;
    std::atomic<int> b(0);
    std::atomic<int> lock(0);
#pragma omp parallel 
    {
      sync(lock);
      a++;
      b.fetch_add(1,std::memory_order_acq_rel);;
    }
    
    std::cout << lock << " " << a << ' ' << b << std::endl;
    
    a=0; b=0;
    
#pragma omp parallel 
    {
      a++;
      b.fetch_add(1,std::memory_order_acq_rel);
    }

    
    std::cout << lock << " " << a << ' ' << b << std::endl;
  }

  return 0;
}
