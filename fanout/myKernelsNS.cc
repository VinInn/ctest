//---------------  this in a file for each API  (say  mykernels.cc mykernels.cu mykernels_hip.cc  )
// API is posix, cuda, hip  etc


// #include "API_defines.h"

#define __API__ posixN

#include "KernelFanout.h"


namespace posixN {
template<typename F, typename Tuple, std::size_t... Is>
void callIt(F f, launchParam const & p, const Tuple& t,std::index_sequence<Is...>){
  f(std::get<Is>(t)...);
}
}



// #include "mykernels.h"
#include <cstdio>

namespace {
void foo(int a, float * b, float * c) {
   printf("%d\n",a);
}
}

DefineWrapper(foo,int,float*,float*)


// dummyes
namespace cudaN {
void CAT(foo,Wrapper)(launchParam const &, std::tuple<int,float*,float*> const&){}
}
namespace hipN {
void CAT(foo,Wrapper)(launchParam const &, std::tuple<int,float*,float*> const &){}
}
