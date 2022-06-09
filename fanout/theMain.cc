#include "KernelFanout.h"


DeclareWrapper(do1, float)
DeclareWrapper(do2, double)

void bha(float x, double y, API api) {
  launchParam p{api}; int n=4;
  launchKernelWrapper(do1,p,x);
  launchKernelWrapper(do2,p,y);
}



int main() {
  bha(1.,-1.,API::posix);

  return 0;
}
