#include<iostream>
#include <dlfcn.h>

void fooWrapper();
void docheck();
int main() {

  fooWrapper();
  dlopen("libdynCrash.so",RTLD_LAZY|RTLD_GLOBAL);
  docheck();
  return 0;
}
