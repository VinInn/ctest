#include <cstdlib>
#include <malloc.h>
#include <libc/malloc.h>
#include <iostream>


extern "C" 
void myMallocHook(size_t size, void *block) {
  std::cout << "asked " << size 
	    << " allocated " << (BLOCK *)block->size
	    << std::endl;
}


int main() {

  int * a = new int[10];

  __libc_malloc_hook = myMallocHook;
  a = new int[10];
 
 __libc_malloc_hook = 0;
  a = new int[10];
 
}
