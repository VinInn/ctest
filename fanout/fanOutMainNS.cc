enum API {posix,cuda,hip};

struct launchParam{API api;};


#define CAT(X,Y) X ## Y
#define launchKernelWrapper(FOO, param, ...) \
  switch (param.api) { \
    case API::posix : \
      posixN::CAT(FOO,Wrapper)(param, __VA_ARGS__);break;\
   case API::cuda : \
      cudaN::CAT(FOO,Wrapper)(param, __VA_ARGS__);break;\
   case API::hip : \
      hipN::CAT(FOO,Wrapper)(param, __VA_ARGS__);break;\
  }


#define DeclareWrapper(FOO, ...) \
namespace posixN { \
void CAT(FOO,Wrapper)(launchParam const &, __VA_ARGS__);\
}\
namespace cudaN { \
void CAT(FOO,Wrapper)(launchParam const &, __VA_ARGS__);\
}\
namespace hipN { \
void CAT(FOO,Wrapper)(launchParam const &, __VA_ARGS__);\
}


DeclareWrapper(foo, int, float *, float *);

void bha(float * x, float * y, API api) {
  launchParam p{api}; int n=4;  
  launchKernelWrapper(foo,p,n,x,y);
}



int main() {
  bha(0,0,API::posix);

  return 0;
}
