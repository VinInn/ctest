//  cat /proc/meminfo | grep -i anon
//  ps -eo pid,command,rss,vsz | grep a.out
// strace -e trace=memory ./a.out
#include<iostream>
#include<vector>
#include<memory>
#include<chrono>

#include "memory_usage.h"

auto start = std::chrono::high_resolution_clock::now();

void stop(const char * m) {
  auto delta = std::chrono::high_resolution_clock::now()-start;
  std::cout << m;
  std::cout << " elapsted time " << std::chrono::duration_cast<std::chrono::nanoseconds>(delta).count() << std::endl;
  std::cout << " allocated so far " << memory_usage::allocated();
  std::cout << " deallocated so far " << memory_usage::deallocated() << std::endl;
  std::cout << "total live " << memory_usage::totlive() << std::endl;
  char c;
  std::cin >> c;

  start = std::chrono::high_resolution_clock::now();
}



int main() {

  std::cout << "jemalloc counters are " << (memory_usage::is_available() ? "" : "NOT ") << "available" << std::endl;

  stop("start");

  constexpr size_t N = 1000*1000*1000;

{
//    auto v = std::make_unique<int[]>(N);
    auto v = new int[N];
    stop("after create");
    v[0]=1;
    stop("after assign 0");
    for (int i=0; i<N; i+=100000) v[i]=1;
    stop("after assign many");

    delete [] v;

  }
  stop("after first block");
 {
    std::vector<int> v;
    v.reserve(N);
    stop("after reserve");
    v.resize(N);
    stop("after resize");
    v[0]=1;
    stop("after assign 0");
    for (int i=0; i<N; i+=10000) v[i]=1;
    stop("after assign many");

  }

  stop("stop");

  return 0;
}
