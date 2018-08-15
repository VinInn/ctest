#include <stdio.h>


__global__
void perm(int pad[]) {
    int t = threadIdx.x;
    int n=0;
    __shared__ bool done;

    pad[2] =0;

    done=true;
    __syncthreads();
    while (done) {
      done = false;
      __syncthreads();
      if ( t == 1 ) {
        ++n;
        pad[0] = 112211;
        if (n<10) done = true;
      }
     __syncthreads();
    }

    __syncthreads();

      if ( t == 17 ) {
          pad[1] = pad[0];
      }

 
   
   pad[2] = 321321321;
}


int main() {

    int h_pad[3];
    int *dev_pad = 0;
    cudaMalloc(&dev_pad, sizeof(h_pad));
    cudaMemset(dev_pad, 0, sizeof(h_pad));

    perm<<< 1, 256 >>>(dev_pad);

    cudaMemcpy(h_pad, dev_pad, sizeof(h_pad), cudaMemcpyDeviceToHost);


    printf("pad[0] = %d    pad[1] = %d  pad[2] = %d\n", h_pad[0], h_pad[1], h_pad[2]);
    return 0;

}

