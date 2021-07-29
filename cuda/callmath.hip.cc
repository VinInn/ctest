#include <hip/hip_runtime.h>
#include<cmath>

__global__ void doit(float x, float * y) {
   *y = acoshf(x);
  // *y = sqrtf(x);

}


__global__ void doitD(double x, double * y) {
  *y = acosh(x);
 //  *y = sqrt(x);
 //  *y = __dsqrt_rn(x);

}

