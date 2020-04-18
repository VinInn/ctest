#include "emptyKernel.h"

#include <iostream>

void c() {
  int ptx_version = 0;
  if (PtxVersion(ptx_version)==cudaSuccess) std::cout << "ptx version is " << ptx_version << std::endl;
  else std::cout << "Error" << std::endl;

}
