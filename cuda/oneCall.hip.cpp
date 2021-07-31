#include <hip/hip_runtime.h>
#include<cmath>
#include<cstdlib>

__global__ void doit(float x) {
   auto y =y1f(x);
   printf ("float %a %a\n",x,y);
}

__global__ void doit(double x) {
   auto y =y1(x);
   printf ("double %a %a\n",x,y);
}


int main() {

  float f = -0.0f;
  hipLaunchKernelGGL(doit,dim3(1),dim3(1),0,0,f);
  auto y =y1f(f);
  printf ("cpu float %a %a\n",f,y);
  double    d = -0.0f;
  hipLaunchKernelGGL(doit,dim3(1),dim3(1),0,0,d);
  hipStreamSynchronize(0);
}
