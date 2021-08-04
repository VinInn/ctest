#include <hip/hip_runtime.h>
#include<cmath>
#include<cstdlib>

__global__ void doit(float x) {
   auto y =log(x);
   printf ("float %a %a\n",x,y);
}

__global__ void doit(double x) {
   auto y =log(x);
   auto z1= lgamma(x);
   auto z2= lgamma(-x);
   printf ("double %a %a %a %a\n",x,y,z1,z2);
}


int main() {

  float f = 0x1.5efad5491a79bp-1022;
  hipLaunchKernelGGL(doit,dim3(1),dim3(1),0,0,f);
  auto y =log(f);
  printf ("cpu float %a %a\n",f,y);
  double    d =  0x1.p500*std::numeric_limits<double>::min();  //  0x1.0p-9; // 0x1.5efad5491a79bp-1022;
  hipLaunchKernelGGL(doit,dim3(1),dim3(1),0,0,d);
  hipStreamSynchronize(0);
}
