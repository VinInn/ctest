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
