#include <stdio.h>
#include<algorithm>

__global__
void perm(int * pad, int nt) {
    int off = blockIdx.x*(3+256+4);
    int n=0;

    pad +=off;
    pad[2] =0;
    for (int t = threadIdx.x; t < nt; t += blockDim.x) pad[3+t] = t%64 + 2;
    pad[260]=0;
    __syncthreads();

    bool more=true;
    while (__syncthreads_or(more)) {
      more = false;
      for (int t = threadIdx.x; t < nt; t += blockDim.x) {
       if(t%15==0) continue;
      for (int j=std::max(0,int(t)-20); j<t; ++j) {
         if(j%15==0) continue;
        // if (3==t%4) { pad[3+t]++; }
        if ( t == 5 ) {
          ++n;
          pad[0] = 112211;
//          if (n<10) more = true;
        }
//        if (2==t%2) {
          auto old = atomicMin(&pad[3+j],pad[3+t]);
          if(old!=pad[3+t]) more=true;
          atomicMin(&pad[3+t],old);
//        }
      }}
      if(0==threadIdx.x) ++pad[260];
    }

    __syncthreads();

      if ( threadIdx.x == 17 ) {
          pad[1] = pad[0];
      }

 
   
   pad[2] = 321321321;
}


int main() {

    int h_pad[2000*(3+256+4)];
    int *dev_pad = 0;
    cudaMalloc(&dev_pad, sizeof(h_pad));
    cudaMemset(dev_pad, 0, sizeof(h_pad));

    printf("size %d\n",sizeof(h_pad));

    perm<<< 2000, 512 >>>(dev_pad,232);

    cudaMemcpy(h_pad, dev_pad, sizeof(h_pad), cudaMemcpyDeviceToHost);


    printf("pad[0] = %d    pad[1] = %d  pad[2] = %d pad[last] = %d \n", h_pad[0], h_pad[1], h_pad[2], h_pad[2+256]);
    printf("loops %d\n",h_pad[260]); 
    return 0;

}

