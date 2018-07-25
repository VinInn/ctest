#include <stdio.h>


__global__
void perm(int pad[]) {
    int t = threadIdx.x;
    int dxN;

    pad[2] =0;

    if (t==11) return;

    if ( t == 1 ) {
        pad[0] = 112211;
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

