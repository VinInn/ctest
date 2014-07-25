#include "Base.h"
#include <dlfcn.h>
#include <cassert>
#include <atomic>
#include<iostream>

Base * Base::me=nullptr;
std::atomic<int> Base::a(0);
__thread int Base::b=0;

int main() {

 std::cout << "one" << std::endl;

 void * dl = dlopen("./threads.so",RTLD_LAZY);
 assert(dl!=nullptr);

 std::cout << "one" << std::endl;


#pragma omp parallel for
for (int i=0; i<100; ++i) {
  Base::a++; Base::b++;
}

 std::cout << "one" << std::endl;
 std::cout << Base::a << std::endl;
 std::cout << Base::b << std::endl;

  dl = dlopen("./tls.so",RTLD_LAZY);
  assert(dl!=nullptr);
  assert(Base::me!=nullptr);

 std::cout << "one" << std::endl;

  return 0;
}    

