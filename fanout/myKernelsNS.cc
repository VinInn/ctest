//---------------  this in a file for each API  (say  mykernels.cc mykernels.cu mykernels_hip.cc  )
// API is posix, cuda, hip  etc

enum API {posix,cuda,hip};

struct launchParam{API api;};


// #include "API_defines.h"

#define __API__ posixN
#define launchKernel(FOO, param, ...) \
  foo(__VA_ARGS__);



// #include "mykernels.h"
#include <cstdio>
void foo(int a, float * b, float * c) {
   printf("%d\n",a);
}



namespace __API__ {
void fooWrapper(launchParam const & p, int a, float * b, float * c) {
   launchKernel(foo,p,a,b,c);
}
}


// dummyes
namespace cudaN {
void fooWrapper(launchParam const & p, int a, float * b, float * c) {
}
}
namespace hipN {
void fooWrapper(launchParam const & p, int a, float * b, float * c) {
}
}
