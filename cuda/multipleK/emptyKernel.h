#pragma once

#include "cuda.h"



template <typename T>
__global__ void EmptyKernel(void) { }


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
