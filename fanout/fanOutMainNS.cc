#include "KernelFanout.h"


DeclareWrapper(foo, int, float *, float *)

void bha(float * x, float * y, API api) {
  launchParam p{api}; int n=4;  
  launchKernelWrapper(foo,p,n,x,y);
}



int main() {
  bha(0,0,API::posix);

  return 0;
}
