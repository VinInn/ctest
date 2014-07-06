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


inline
void runparallel() {

  static __thread bool first=true;
  if (first)
  {
    first=false;
    Lock l(global::coutLock);

    std::cout << "thread " << omp_get_thread_num() << " of " << omp_get_num_threads() << std::endl;

    if(omp_in_parallel()) std::cout << "in parallel" << std::endl;

  }

}



struct Go {



  void doparallel() {
    runparallel();

  }

  void evaluate() {
    for (int i=0; i!=100000; ++i) {
    #pragma omp parallel
    {

    doparallel();

    }
    }
   std::cout << "done" << std::endl;
  }

};

int main()  {

  double * data = new double[800000];


#pragma omp parallel
{

  runparallel();

}


  Go go; go.evaluate();

  delete data;

  return 0;

}
