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

inline
void runparallel(int k ) {
  {
    Lock l(global::coutLock);

    std::cout << k << " thread " << omp_get_thread_num() << " of " << omp_get_num_threads() << std::endl;
  }

  //  static __thread bool first=true;
  static thread_local bool first=true;
  if (first)
  {
    first=false;
    Lock l(global::coutLock);

    std::cout << "first time thread " << omp_get_thread_num() << " of " << omp_get_num_threads() << std::endl;

    if(omp_in_parallel()) std::cout << "in parallel" << std::endl;

  }

}

int main()  {

for (int i=0; i!=10; ++i) {
#pragma omp parallel
{
  runparallel(i);
}
}

return 0;

}

