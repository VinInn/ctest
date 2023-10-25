#include <stacktrace>
#include<iostream>

  std::string get_stacktrace() {
     std::string trace;
     for (auto & entry : std::stacktrace::current() ) trace += entry.description() + '#';
     return trace;
  }



inline
void print_stacktrace() {
  std::cout << std::stacktrace::current() << std::endl;
}


#include <cstdlib>
#include <malloc.h>
#include <iostream>

extern "C"
void * myMallocHook(size_t size, void const * caller) {
  __malloc_hook = nullptr;
  auto p = malloc(size);
  std::cout << "asked " << size
            << " at " << get_stacktrace() << std::endl;
//  print_stacktrace(); 
  __malloc_hook = myMallocHook;
  return p;
}

namespace {
struct Hook {
  Hook() {
  __malloc_hook = myMallocHook;
  }
};

  Hook hook;

}
