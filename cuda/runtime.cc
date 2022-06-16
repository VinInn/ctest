// c++ -O3 runtime.cc -I/opt/rocm-5.1.1/include -D__HIP_PLATFORM_AMD__ -L/opt/rocm-5.1.1/lib -lamdhip64
// c++ -O3 runtime.cc -I/opt/rocm-5.1.1/include -D__HIP_PLATFORM_NVIDIA__ -L/opt/rocm-5.1.1/lib -lamdhip64 -I/usr/local/cuda-11.6/include/ -L/usr/local/cuda-11.6/lib64/ -lcudart
// setenv LD_LIBRARY_PATH /opt/rocm-5.1.1/lib:${LD_LIBRARY_PATH}

#include "hip/hip_runtime.h"
#include <hip/hip_runtime_api.h>

#include<iostream>



void go() {

   int cuda_device = 0;
   hipDeviceProp_t deviceProp;
   auto err = hipGetDeviceProperties(&deviceProp, cuda_device);
   printf("Device: %s, SM %d.%d hardware\n", deviceProp.name, deviceProp.major, deviceProp.minor);

   hipStream_t stream;

   err = hipStreamCreate(&stream);
   
  err = hipStreamSynchronize(stream);
  if (err != hipSuccess)  std::cout << "got an error " << hipGetErrorName(err) << ' ' << hipGetErrorString(err) << std::endl;

}


int main() {

   go();

}

