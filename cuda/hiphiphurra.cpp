#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>



__global__ void doit() {

  printf("hurra %f\n", std::sinf(0.45f));


}



#include<iostream>
int main() {

#ifdef __HIP_PLATFORM_AMD__
  std::cout << "on a AMD GPU" << std::endl;
#endif


hipDeviceProp_t props;

    int deviceIndex =0;

    // Check device index is in range
    int count;
    hipGetDeviceCount(&count);
    // check_cuda_error();
    if (deviceIndex >= count)
        throw std::runtime_error("Chosen device index is invalid");
    hipSetDevice(deviceIndex);

    hipGetDeviceProperties(&props, deviceIndex);

    // Print out device name
    std::cout << "Using HIP device " <<  " (compute_units=" << props.multiProcessorCount << ")" << std::endl;

    // Print out device HIP driver version
    // std::cout << "Driver: " << getDriver() << std::endl;


  hipLaunchKernelGGL(doit,dim3(1),dim3(1),0,0);
//  std::cout << "launching with error " < err < std::endl;
   auto err = hipDeviceSynchronize();
   std::cout << "launching with error " << err << std::endl;
  return 0;

}
