#include<cuda.h>
#include<cuda_runtime.h>
#include <cuda_runtime_api.h>
#include<iostream>


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



typedef union { unsigned int n; float x; } union_t;

constexpr int maxNumOfThreads = 24;

const int bunchSize = 1024;

#include<cmath>
__global__ void kernel_foo(unsigned int n, float * py) {
   int first = blockIdx.x * blockDim.x + threadIdx.x;
   for (int i=first; i<bunchSize; i+=gridDim.x*blockDim.x) {
     union_t u; u.n = n+i; float x = u.x;
     py[i] = std::sin(x);
   }
}


void cpu_foo(unsigned int n, float * py) {
   int first = 0;
   for (int i=first; i<bunchSize; i++) {
     union_t u; u.n = n+i; float x = u.x;
     py[i] = std::sin(x);
   }
}

template <typename F>
void CUDART_CB MyCallback(void * fun){
    (*(F*)(fun))();
}


void CUDART_CB aCallback(void *data){
    printf("Inside callback %d\n", (size_t)data);
}

void compare(float * yd, float * yh, float & dm) {
   int first = 0;
   for (int i=first; i<bunchSize; i++) {
     auto d = std::abs(yd[i]-yh[i]);
     dm = std::max(dm,d);
   }
}

#include<atomic>

cudaStream_t streams[maxNumOfThreads];

std::atomic<int> nt(0);

void go() {
  int me = nt++;
  auto & stream = streams[me];

  float * ypD;
  float * ypH;
  float * ypC;
  float dm=0;

  cudaCheck(cudaMalloc((void **)&ypD, bunchSize*sizeof(float)));
  cudaCheck(cudaMallocHost((void **)&ypH, bunchSize*sizeof(float)));
  ypC =(float*)::malloc(bunchSize*sizeof(float));

  kernel_foo<<<1024/128,128,0,stream>>>(me*bunchSize, ypD);

  auto k1 = [&]() {
    cpu_foo(me*bunchSize, ypC);
  };

  // auto f1 = MyCallback<k1>;

  auto k2 = [&]() {
    compare(ypH,ypC,dm);
  };

  cudaCheck(cudaMemcpyAsync(ypH, ypD, bunchSize*sizeof(float), cudaMemcpyDeviceToHost, stream));
  cudaLaunchHostFunc (stream, aCallback, &me);
  cudaLaunchHostFunc (stream, MyCallback<decltype(k1)>, &k1);
  cudaLaunchHostFunc (stream, MyCallback<decltype(k2)>, &k2);


  cudaStreamSynchronize(stream);

  std::cout << "max diff in " << me << ' ' << dm << std::endl;

}


int main (int argc, char *argv[]) {

  printf ("Using CUDA %d\n",CUDART_VERSION);
  int cuda_device = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, cuda_device);
  printf("CUDA Capable: SM %d.%d hardware\n", deviceProp.major, deviceProp.minor);


  int nstreams = maxNumOfThreads;

  cudaStream_t streams[maxNumOfThreads];


  for (int i = 0; i < nstreams; i++) {
        cudaCheck(cudaStreamCreate(&(streams[i])));
  }

  go();


  return 0;

}
