#include <cassert>
struct Go {

  virtual ~Go(){}

  virtual void go() =0;

  inline
  static Go * me(Go * in=nullptr) {
    static Go * l = nullptr;
    if (in) l=in;
    return l;
  }

};


#include<iostream>
#include <dlfcn.h>

void fooWrapper();
void docheck();
int main() {

  fooWrapper();
  assert(!Go::me());
  dlopen("libdynCrash.so",RTLD_LAZY|RTLD_GLOBAL);
  assert(Go::me());
  Go::me()->go();
  docheck();
  return 0;
}
