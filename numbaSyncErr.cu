#include <stdio.h>


__global__
void perm(int pad[]) {
    int t = threadIdx.x;
    int dxN;

    if ( t >= 0 ) {
        dxN = pad[0];
    }

    if ( t < 14 ) {
        if ( t >= 0 ) {
            // The following branch is the reason
            // comment it out to get the correct behavior
            if (dxN + 1 == 0) goto ERROR;
            pad[2] = 5 / (dxN + 1);
        }
    }

    if ( t == 1 ) {
        pad[0] = 112211;
    }

    __syncthreads();

    if ( t == 17 ) {
        pad[1] = pad[0];
    }

ERROR:
    pad[2] = 321321321;
}


int main() {

    int h_pad[3];
    int *dev_pad = 0;
    cudaMalloc(&dev_pad, sizeof(h_pad));
    cudaMemset(dev_pad, 0, sizeof(h_pad));

    perm<<< 1, 20 >>>(dev_pad);

    cudaMemcpy(h_pad, dev_pad, sizeof(h_pad), cudaMemcpyDeviceToHost);


    printf("pad[0] = %d    pad[1] = %d\n", h_pad[0], h_pad[1]);
    return 0;

}

