#include <cstdio>
#include <cstdint>
#include <algorithm>


__global__
void perm(uint16_t const __restrict__ * x, int * id,  int nt) {

    if (threadIdx.x==0) id[nt]=0;
    for (int t = threadIdx.x; t < nt; t += blockDim.x) id[t]=t;
    __syncthreads();

    bool more=true;
    while (__syncthreads_or(more)) {
      more = false;
      for (int t = threadIdx.x; t < nt; t += blockDim.x) {
         for (auto m = t+1; m<nt; ++m) {
          if (std::abs(x[m]-x[t])>1) continue;
          auto old = atomicMin(&id[m],id[t]);
          if(old!=id[t]) more=true;
          atomicMin(&id[t],old);
        }
      }
      if (threadIdx.x==0)  ++id[nt];
    }

    __syncthreads();
}


#include<iostream>
int main() {

    uint16_t x[1024]; 
    int id[1024];

    uint16_t * x_d;
    int * id_d;
    cudaMalloc(&x_d, sizeof(x));
    cudaMalloc(&id_d, sizeof(id));

    x[0]=1;x[1]=1;x[2]=1;

    int n =3;

    cudaMemcpy(x_d, x, sizeof(x),cudaMemcpyHostToDevice);

    printf("size %d\n",sizeof(x));

    perm<<< 1, 256 >>>(x_d,id_d,n);

    cudaMemcpy(id, id_d, sizeof(x),cudaMemcpyDeviceToHost);

    std::cout << "ids " << id[0] << ' ' << id[n-1] << " loops " << id[n] << std::endl;

    return 0;

}

