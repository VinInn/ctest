#include <cstdint>
#include <dlfcn.h>
#include <iostream>
#include<cassert>

namespace {

  thread_local bool notRecording = true;

/*
struct  Me {
  Me() {
    // setenv("LD_PRELOAD","", true);
    std::cout << "Recoding structure constructed in a thread " << std::endl;
    notRecording = true;
  }

 
};

  thread_local Me * me = new Me();
*/


}

extern "C" 
{


/*
void* malloc(size_t size)
{
    typedef void * (*m) (size_t);
    // static void* (*real_malloc)(size_t) = NULL;
    // if (!real_malloc)
    static auto real_malloc = (m)dlsym(RTLD_NEXT, "malloc");

    void *p = real_malloc(size);
    fprintf(stderr, "malloc(%ld) = %p\n", size, p);
    if (notRecording) { fprintf(stderr,"not recording\n"); } 
    else  { fprintf(stderr,"recording\n"); }
    return p;
}
*/


void *malloc(std::size_t size) {
  typedef void * (*m) (std::size_t);
  static auto origf = (m)dlsym(RTLD_NEXT,"malloc");
  assert(origf);
  auto p  = origf(size);
  if (notRecording) {
    notRecording = false;
    
    std::cout << "m " << size << ' ' << p << std::endl;
    
    notRecording = true;
  }
  return p;
}




void free(void *ptr) {
  typedef void (*m) (void*);
  static auto origf = (m)dlsym(RTLD_NEXT,"free");
  assert(origf);
  if (notRecording) {
    notRecording = false;

    std::cout << "f " << ptr << std::endl;
    notRecording = true;
  }

  origf(ptr);

}


}

