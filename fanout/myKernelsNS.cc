//---------------  this in a file for each API  (say  mykernels.cc mykernels.cu mykernels_hip.cc  )
// API is posix, cuda, hip  etc

enum API {posix,cuda,hip};

struct launchParam{API api;};


// #include "API_defines.h"

#define __API__ posixN


#include<tuple>
#include<utility>
namespace posixN {
template<typename F, typename Tuple, std::size_t... Is>
void callIt(F f, launchParam const & p, const Tuple& t,std::index_sequence<Is...>){
  f(std::get<Is>(t)...);
}
}


#define CAT(X,Y) X ## Y
#define DefineWrapper(FOO, ...) \
namespace __API__ { \
void CAT(FOO,Wrapper)(launchParam const & p, std::tuple<__VA_ARGS__> const & t){\
   callIt(FOO, p, t, std::index_sequence_for<__VA_ARGS__>()); \
}\
}



// #include "mykernels.h"
#include <cstdio>
void foo(int a, float * b, float * c) {
   printf("%d\n",a);
}


DefineWrapper(foo,int,float*,float*)


// dummyes
namespace cudaN {
void CAT(foo,Wrapper)(launchParam const &, std::tuple<int,float*,float*> const&){}
}
namespace hipN {
void CAT(foo,Wrapper)(launchParam const &, std::tuple<int,float*,float*> const &){}
}
