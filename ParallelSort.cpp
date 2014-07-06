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

#include "future.h"

template<typename F>
std::unique_future<typename std::result_of<F()>::type> spawn_task(F f) {
  typedef typename std::result_of<F()>::type result_type;
  std::packaged_task<result_type()> task(std::move(f));
  std::unique_future<result_type> res(task.get_future());
  std::thread t(std::move(task));
  t.detach();
  return res;
}



size_t SORT_THRESHOLD = 100000;

/* re-entrant function
 * Mattson el al. 5.29 page 170
 */
template<typename Iter, typename Compare>
void parallel_sort(Iter b, Iter e, Compare c) {
  size_t n = std::distance(b,e);

  // final exit
  if (n< SORT_THRESHOLD) return std::sort(b,e,c);

  // Divide
  Iter pivot = b +n/2;

  // Conquer

  // fork first half
  //Thread forked(parallel_sort<Iter,Compare>,b,pivot,c);

  auto res = spawn_task(std::bind(parallel_sort<Iter,Compare>,b,pivot,c));

  // serial version
  // parallel_sort(b, e,c);

  // process locally second half
  parallel_sort(pivot,e,c);

  // wait for the other half
  // forked.join();
  res.wait();

  // merge...
  std::inplace_merge(b,pivot,e);
		 

}


int main() {

  size_t SIZE = 10000000;

  std::vector<int> v(SIZE);

  std::generate(v.begin(),v.end(),std::rand);

  parallel_sort(v.begin(),v.end(), std::less<int>());

  for (int i=1; i<SIZE; ++i)
    if (v[i]<v[i-1]) std::cout << "error at " << i << std::endl;




  return 0;
}
