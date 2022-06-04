enum API {posix,cuda,hip};

struct launchParam{API api;};


#define CAT(X,Y) X ## Y
#define launchKernelWrapper(FOO, param, ...) \
  switch (param.api) { \
    case API::posix : \
      posixN::CAT(FOO,Wrapper)(param, {__VA_ARGS__});break;\
   case API::cuda : \
      cudaN::CAT(FOO,Wrapper)(param, {__VA_ARGS__});break;\
   case API::hip : \
      hipN::CAT(FOO,Wrapper)(param, {__VA_ARGS__});break;\
  }

#include<tuple>
#include<utility>
template<typename F, typename Tuple, std::size_t... Is>
void callIt(F f,const Tuple& t,std::index_sequence<Is...>){
  f(std::get<Is>(t)...);
}


using ww = std::tuple<int, float *, float *>;

#define DeclareWrapper(FOO, ...) \
namespace posixN { \
void CAT(FOO,Wrapper)(launchParam const &, std::tuple<__VA_ARGS__> const&);\
}\
namespace cudaN { \
void CAT(FOO,Wrapper)(launchParam const &, std::tuple<__VA_ARGS__> const&);\
}\
namespace hipN { \
void CAT(FOO,Wrapper)(launchParam const &, std::tuple<__VA_ARGS__> const &);\
}


DeclareWrapper(foo, int, float *, float *)

void bha(float * x, float * y, API api) {
  launchParam p{api}; int n=4;  
  launchKernelWrapper(foo,p,n,x,y);
}



int main() {
  bha(0,0,API::posix);

  return 0;
}
