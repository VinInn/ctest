#include <cstdlib>
#include <iostream>

// if cuda is included remove ifdef
#ifdef __CUDACC__
#include <cuda_runtime.h>

void exitSansCUDADevices() {
  int devices = 0;
  auto status = cudaGetDeviceCount(& devices);
  if (status != cudaSuccess) {
    std::cerr << "Failed to initialise the CUDA runtime, the test will be skipped." << "\n";
    exit(EXIT_SUCCESS);
  }
  if (devices == 0) {
    std::cerr << "No CUDA devices available, the test will be skipped." << "\n";
    exit(EXIT_SUCCESS);
  }
}
#else
void exitSansCUDADevices(){ 
    std::cerr << "No CUDA devices available, the test will be skipped." << "\n";
    exit(EXIT_SUCCESS);
}
#endif
