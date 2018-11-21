#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"
#include <iostream>
#include <vector>
#include <atomic>
#include <random>
#include <algorithm>
#include <type_traits>


int main() {


  constexpr uint32_t n = 17*40000;
  constexpr uint32_t m = n*20;

  std::vector<float> res(n,0);
  std::vector<float> q(m);

  int ok=0;
  for(auto & r: q) { 
    r = (++ok ==2) ? 1 : -1.f;
    if (ok==7) ok=0;
  }

  constexpr float c = 3.14;

  auto theLoop = [&](int i) {
    auto nn = n;
    for (int j=0; j<nn; ++j)
      res[j] =  (q[j+i*nn]>0) ? c : res[j];
     // if (q[j+i*nn]>0) res[j]=c;
  };


  tbb::parallel_for(
            tbb::blocked_range<size_t>(0,10),
            [&](const tbb::blocked_range<size_t>& r) {
              for (size_t i=r.begin();i<r.end();++i) theLoop(i);
    }
  );

  tbb::parallel_for(
            tbb::blocked_range<size_t>(0,20),
            [&](const tbb::blocked_range<size_t>& r) {
              for (size_t i=r.begin();i<r.end();++i) theLoop(i);
    }
  );

  std::cout << std::count(res.begin(),res.end(),c) << std::endl;



   return 0;

}
