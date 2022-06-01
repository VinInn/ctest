enum API {posix,cuda,hip};

struct launchParam{API api;};


#define CAT(X,Y) X ## Y
#define launchKernelWrapper(FOO, param, ...) \
  switch (param.api) { \
    case API::posix : \
      CAT(FOO,Wrapper<API::posix>)(param, __VA_ARGS__);break;\
   case API::cuda : \
      CAT(FOO,Wrapper<API::cuda>)(param, __VA_ARGS__);break;\
   case API::hip : \
      CAT(FOO,Wrapper<API::hip>)(param, __VA_ARGS__);break;\
  }

template<API api>
void fooWrapper(launchParam const &, int a, float * b, float * c);

void bha(float * x, float * y, API api) {
  launchParam p{api}; int n=4;  
  launchKernelWrapper(foo,p,n,x,y);

}


int main() {
  bha(0,0,API::posix);

  return 0;
}
