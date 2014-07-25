#include "Base.h"
#include <dlfcn.h>
#include <cassert>
#include <atomic>

Base * Base::me=nullptr;
std::atomic<int> Base::a(0);


int main() {

#pragma omp parallel for
for (int i=0; i<100; ++i) {
  Base::a++;
}


  void * dl = dlopen("./tls.so",RTLD_LAZY);
  assert(dl!=nullptr);
  assert(Base::me!=nullptr);

  return 0;
}    

