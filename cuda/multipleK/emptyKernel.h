#pragma once

#include "cuda.h"



template <typename T>
#ifndef __NVCC__ 
__attribute__((weak))
#endif
__global__ void EmptyKernel(void) {

#ifdef TEST_MACRO

#ifndef __CUDA_ARCH__
  #warning in kernel inside "ifndef  __CUDA_ARCH__"
#endif

#ifndef __CUDA__
  #warning in kernel inside "ifndef  __CUDA__"
#endif

#endif

}


inline __attribute__((always_inline)) cudaError_t PtxVersion(int &ptx_version)
{
    struct Dummy
    {
        /// Type definition of the EmptyKernel kernel entry point
        typedef void (*EmptyKernelPtr)();

        /// Force EmptyKernel<void> to be generated if this class is used
        inline __attribute__((always_inline))
        EmptyKernelPtr Empty()
        {
            return EmptyKernel<void>;
        }
    };

    cudaError_t error = cudaSuccess;
    cudaFuncAttributes empty_kernel_attrs;
    error = cudaFuncGetAttributes(&empty_kernel_attrs, EmptyKernel<void>) ;
    if(!error) ptx_version = empty_kernel_attrs.ptxVersion * 10;

    return error;
}
