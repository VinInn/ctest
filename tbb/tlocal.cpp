#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"
#include <iostream>
#include <vector>
#include <atomic>
#include <random>
#include <algorithm>
#include <type_traits>
#include<cassert>

thread_local int tl;

struct Init {
  Init() { tl=1;}
};

const Init bha;


std::atomic<int> c;
void count() {
  c++;
}

void verifyInit() {
  assert(tl==1);
}

void verifyloop(int i) {
  assert(tl==i);
}


int main() {

  c=0;
  tbb::parallel_for(
            tbb::blocked_range<size_t>(0,1000),
            [&](const tbb::blocked_range<size_t>& r) {
              count();
              for (size_t i=r.begin();i<r.end();++i) verifyInit();
    }
  );

  std::cout << "count init " << c << std::endl;

  c=0;
  tbb::parallel_for(
            tbb::blocked_range<size_t>(0,1000),
            [&](const tbb::blocked_range<size_t>& r) {
              count();
              for (size_t i=r.begin();i<r.end();++i) {
                tl=i; verifyloop(i);
              }
    }
  );

  std::cout << "count loop " << c << std::endl;



};
