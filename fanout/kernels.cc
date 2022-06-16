#include "API.h"
#include<cmath>
#include<cstdio>


namespace  {
__global__ void do1(float x) {
   printf ("cos(%f) =  %f\n",x, cosf(x));
}


__global__ void do2(double x) {
   printf ("sin(%f) =  %f\n",x, sin(x));
}
}

DefineWrapper(do1,float)
DefineWrapper(do2,double)

