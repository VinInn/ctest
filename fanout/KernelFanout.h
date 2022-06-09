#pragma once

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



#define DefineWrapper(FOO, ...) \
namespace __API__ { \
void CAT(FOO,Wrapper)(launchParam const & p, std::tuple<__VA_ARGS__> const & t){\
   callIt(FOO, p, t, std::index_sequence_for<__VA_ARGS__>()); \
}\
}

