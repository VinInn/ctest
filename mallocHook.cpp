// compile with
// c++ -O3 -pthread -fPIC -shared -std=c++23 getStacktrace.cc /data/user/innocent/gcc_build/x86_64-pc-linux-gnu/libstdc++-v3/src/libbacktrace/.libs/libstdc++_libbacktrace.a -g -o mallocHook.so
// run as
// setenv LD_PRELOAD ./mallocHook.so ; ./a.out ; unsetenv LD_PRELOAD


#include <cstdlib>
#include <malloc.h>
#include <iostream>


extern "C" 
void * myMallocHook(size_t size, void const * caller) {
  __malloc_hook = nullptr;
  auto p = malloc(size);
  std::cout << "asked " << size 
	    << std::endl;
  __malloc_hook = myMallocHook;
  return p;
}


int main() {

  int * a = new int[10];

  __malloc_hook = myMallocHook;
  a = new int[10];

  std::cout << "p " << a << std::endl;
 
 __malloc_hook = nullptr;
  a = new int[10];

  return a[0];
 
}
