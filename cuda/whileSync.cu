#include <stdio.h>


__global__
void perm(int pad[]) {
    int t = threadIdx.x;
    int n=0;

    pad[2] =0;
    pad[3+threadIdx.x] = t;
    pad[260]=0;
    __syncthreads();

    bool more=true;
    while (__syncthreads_or(more)) {
      more = false;
      for (int j=0; j<threadIdx.x; ++j) {
        // if (3==t%4) { pad[3+threadIdx.x]++; }
        if ( t == 5 ) {
          ++n;
          pad[0] = 112211;
          if (n<20) more = true;
        }
        if (2==t%2) {
          auto old = atomicMin(&pad[3+j],pad[3+threadIdx.x]);
          if(old!=pad[3+threadIdx.x]) more=true;
          atomicMin(&pad[3+threadIdx.x],old);
        }
      }
      if(0==threadIdx.x) ++pad[260];
    }

    __syncthreads();

      if ( t == 17 ) {
          pad[1] = pad[0];
      }

 
   
   pad[2] = 321321321;
}


int main() {

    int h_pad[3+256+4];
    int *dev_pad = 0;
    cudaMalloc(&dev_pad, sizeof(h_pad));
    cudaMemset(dev_pad, 0, sizeof(h_pad));

    perm<<< 1, 256 >>>(dev_pad);

    cudaMemcpy(h_pad, dev_pad, sizeof(h_pad), cudaMemcpyDeviceToHost);


    printf("pad[0] = %d    pad[1] = %d  pad[2] = %d pad[last] = %d \n", h_pad[0], h_pad[1], h_pad[2], h_pad[2+256]);
    printf("loops %d\n",h_pad[260]); 
    return 0;

}

