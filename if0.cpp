#include<iostream>

#ifdef VI_DEBUG
#define VIDEBUG_FLAG true
#else
#define VIDEBUG_FLAG false
#endif

int k[10];

void hi() {
  if (VIDEBUG_FLAG) std::cout << "ciao" << std::endl;
}


void foo() {
  if (VIDEBUG_FLAG) {
    for ( auto & i : k) ++i;
  }
 
}
