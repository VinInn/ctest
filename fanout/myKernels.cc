//---------------  this in a file for each API  (say  mykernels.cc mykernels.cu mykernels_hip.cc  )
// API is posix, cuda, hip  etc

enum API {posix,cuda,hip};

struct launchParam{API api;};


// #include "API_defines.h"

#define __API__ posix
#define launchKernel(FOO, param, ...) \
  foo(__VA_ARGS__);


// #include "mykernels.h"
#include <cstdio>
void foo(int a, float * b, float * c) {
   printf("%d\n",a);
}


template<API api, typename ... ARGS>
void fooWrapper(launchParam const &, ARGS... args);

template<>
void fooWrapper<__API__>(launchParam const & p, int a, float * b, float * c) {
   launchKernel(foo,p,a,b,c);
}



// dummyes
template<>
void fooWrapper<API::cuda>(launchParam const & p, int a, float * b, float * c) {
}
template<>
void fooWrapper<API::hip>(launchParam const & p, int a, float * b, float * c) {
}

