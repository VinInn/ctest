#include "tbb/task_group.h"
#include "tbb/task_scheduler_init.h"
#include <iostream>
#include <vector>

struct mytask {
  mytask(size_t n)
    :_n(n)
  {}
  void operator()() const {
    for (int i=0;i<1000000;++i) {}  // Deliberately run slow
    std::cerr << "[" << _n << "]";
  }
  size_t _n;
};

int main(int,char**) {

  std::cerr << "default num of thread " << tbb::task_scheduler_init::default_num_threads() << std::endl;

  //tbb::task_scheduler_init init;  // Automatic number of threads
   tbb::task_scheduler_init init(tbb::task_scheduler_init::default_num_threads());  // Explicit number of threads

   tbb::task_group g;

  auto NTasks = 1000;
  // not necessarely a good idea but works...
  for (auto i=0;i<NTasks;++i)
    g.run(mytask(i));
  g.wait();

  std::cerr << std::endl;
  std::cerr << std::endl;
  std::cerr << std::endl;

  // now with chunks

  auto NChunks = 10;
  for (auto i=0;i<NChunks;++i) {
     auto start = i;
     auto stride = NChunks;
     g.run([=] {
        for (auto j=start; j<NTasks; j+=stride) {
           mytask a(j); a();
        }
     });
  } 
  g.wait();

  std::cerr << std::endl;


  return 0;
}
