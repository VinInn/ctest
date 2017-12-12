#include "tbb/task_group.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/concurrent_queue.h"
#include <iostream>
#include <vector>
#include <memory>
#include <atomic>
#include <cassert>

namespace {

  struct Queue {
    Queue() : ave(0),ncall(0){}
    ~Queue() {
      std::cout << "queue calls, ave-size " << ncall << ' ' << double(ave)/ncall << std::endl;
      for (auto p=me.unsafe_begin(); p!=me.unsafe_end(); ++p) std::cout << *(*p) <<'/';
      std::cout << std::endl;

    }
    using Item = std::unique_ptr<int>;
    tbb::concurrent_queue<Item> me;
    std::atomic<long long> ave;
    std::atomic<long long> ncall;
    auto try_pop(Item & item) {
      ave += me.unsafe_size();
      ncall+=1;
      return me.try_pop(item);
    }

    void push(Item && item) { me.push(std::move(item));}
  
  };

  Queue queue1, queue2;
  
}




int main() {
 
    queue1.push(std::make_unique<int>(1));
    std::unique_ptr<int> a;
    queue1.try_pop(a);
    std::cout << *a  << std::endl;
    std::cout << queue1.me.unsafe_size()  << std::endl;


   std::cerr << "default num of thread " << tbb::task_scheduler_init::default_num_threads() << std::endl;

  //tbb::task_scheduler_init init;  // Automatic number of threads
   tbb::task_scheduler_init init(tbb::task_scheduler_init::default_num_threads());  // Explicit number of threads

   tbb::task_group g;

   queue1.push(std::make_unique<int>(0));
   queue2.push(std::make_unique<int>(0));
 
  auto NTasks = 2000;
  // not necessarely a good idea but works...
  for (auto i=0;i<NTasks;++i) {
    auto k=i;
    g.run([&,k]{
	auto p = k%2 ? &queue1 : &queue2;
	auto & queue = *p;
	std::unique_ptr<int> a;
	if (!queue.try_pop(a))
	  a = std::make_unique<int>(k);
	assert(a.get());
	queue.push(std::move(a));
      }
      );
  }
  g.wait();

  
  return 0;

}
