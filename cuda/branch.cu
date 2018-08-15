#ifdef __NVCC__
#define inline __device__  __host__ inline
#else
#define __global__
#define __host__
#endif

__host__
void bar(float * x, float * r) {
constexpr float tab[2] = { .8f, .1f };
for (int i=0;i<1024;++i)
  r[i] = tab[x[i] > .5f];
}

__host__
void foo(float * x, float * r) {
 for (int i=0;i<1024;++i)
  r[i] = x[i] > .5f ? .1f : .8f;
}


__global__
void
GPU(float * x, float * r) {
  constexpr float tab[2] = { .8f, .1f };
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  r[i] = tab[x[i] > .5f];
}


__global__
void fooGPU(float * x, float * r) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  r[i] = x[i] > .5f ? .1f : .8f;
}
