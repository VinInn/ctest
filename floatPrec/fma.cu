// nvcc -gencode arch=compute_61,code=sm_61 -fmad=false -O3 -ptx fma.cu -o fma.ptx -I/cvmfs/cms.cern.ch/slc7_amd64_gcc630/external/cuda/9.1.85-cms/include ; cat fma.ptx

#include <cmath>

__device__
float myf(float x, float y, float z) {
  return std::fma(x,-y,z);
}

__device__
float myff(float x, float y, float z) {
  return z+x*y;
}

__device__
float myfn(float x, float y, float z) {
  return x*y-z;
}


__device__
float myxyn(float x, float y, float z) {
  return (x*y) - (y*z);
}

__device__
float myxyp(float x, float y, float z) {
  return (x*y) + (y*z);
}



__global__
void go(float * x, float * y, float * z, float * r) {

  r[0] = myf(x[0],y[0],z[0]);

  r[1] = myff(x[1],y[1],z[1]);

  r[2] = myfn(x[2],y[2],z[2]);

  r[3] = myxyn(x[3],y[3],z[3]);

  r[4] = myxyp(x[4],y[4],z[4]);


}
