#include <iostream>
#include <cmath>
#include <cstdio>

#ifdef __CUDACC__
#include<cuda.h>
#include<cuda_runtime.h>
#include <cuda_runtime_api.h>

inline
bool cudaCheck_(const char* file, int line, const char* cmd, cudaError_t result)
{
    //std::cerr << file << ", line " << line << ": " << cmd << std::endl;
    if (result == cudaSuccess)
        return true;

    const char* error = cudaGetErrorName(result);
    const char* message = cudaGetErrorString(result);
    std::cerr << file << ", line " << line << ": " << error << ": " << message << std::endl;
    abort();
    return false;
}
#define cudaCheck(ARG) (cudaCheck_(__FILE__, __LINE__, #ARG, (ARG)))
#endif

__global__ void kernel_foo() {
 volatile float x = 0x1.fffffcp+1; 
 // volatile float x = 0x1.f02102p-13;
 volatile float y = 1.f/sqrtf(x);
 volatile float z = rsqrtf(x);
 volatile float w = __frsqrt_rn(x);
 volatile float d = 1./sqrt(double(x));
 volatile float e = rsqrt(double(x));
 printf ("rsqrt(%a) = %a  %a %a %a %a\n", x, y, z, w, d, e);
}


int
main (int argc, char *argv[])
{
#ifdef __CUDACC__
#ifndef CUDART_VERSION
 #warning "no " CUDART_VERSION
#else
    printf ("Using CUDA %d\n",CUDART_VERSION);
#endif
    int cuda_device = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, cuda_device);
    printf("CUDA Capable: SM %d.%d hardware\n", deviceProp.major, deviceProp.minor);
#endif

    kernel_foo<<<1,1>>>();
    cudaCheck(cudaDeviceSynchronize());


    std::cout << "on CPU" <<  std::endl;
    volatile float x = 0x1.fffffcp+1; // 0x1.f02102p-13;
    // volatile float x = 0x1.f02102p-13;

    volatile float y = 1.f/sqrtf(x);
    volatile float z = rsqrtf(x);
    volatile float w = 1.f/std::sqrt(x);
    volatile float d = 1./sqrt(double(x));
    volatile float e = rsqrt(double(x));
    printf ("rsqrt(%a) = %a  %a %a %a %a\n", x, y, z, w, d, e);

    return 0;

}
