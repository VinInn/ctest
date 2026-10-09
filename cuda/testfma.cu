//
//  /opt/nvidia/nsight-compute/2025.2.1/ncu --metrics sm__sass_thread_inst_executed_op_dfma_pred_on.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,,sm__sass_thread_inst_executed_op_hfma_pred_on.sum ./a.out

#include <cuda_fp16.h>

template<typename T>
__global__ void dofma( T * __restrict__ out, T * const __restrict__  x, T * const __restrict__ y, T * const __restrict__  z, int n) {
   int tid = blockDim.x * blockIdx.x + threadIdx.x;
     if (tid>=n) return;
     for (auto j=tid; j<n; j+=(gridDim.x * blockDim.x))  {
        if constexpr (std::is_same<half2, typename std::remove_cv<T>::type>::value) {
          auto w = __hfma2(out[j],y[j],z[j]);
          out[j] = __hfma2(w,y[j],z[j]);
        } else {
          auto w = out[j]*y[j]+z[j];
          out[j] = w*y[j]+z[j];
        }
     }
}

template<typename T>
__global__ void doOtherOp(T * __restrict__ out, T * const __restrict__  x, T * const __restrict__ y, T * const __restrict__  z, int n) {
     int tid = blockDim.x * blockIdx.x + threadIdx.x;
     if (tid>=n) return;
     for (auto j=tid; j<n; j+=(gridDim.x * blockDim.x))  {
       out[j] = sqrtf(x[j]) + __frcp_rn(y[j]) + x[j]/z[j] + rsqrtf(out[j]);
     }
}

int main() {
 
 double * d;
 float * f;
 half  * h;

 int n = 240*1024;

 cudaMallocManaged(&d, 4*n*sizeof(double));
 cudaMallocManaged(&f, 4*n*sizeof(float));
 cudaMallocManaged(&h, 4*n*sizeof(half));

 half2 * h2 = (half2*)(h);

 dofma<double><<<40,64,0,0>>>(d,d+n,d+2*n,d+3*n,n);
 dofma<float><<<40,64,0,0>>>(f,f+n,f+2*n,f+3*n,n);
 dofma<half><<<40,64,0,0>>>(h,h+n,h+2*n,h+3*n,n);
 dofma<half2><<<40,64,0,0>>>(h2,h2+n/2,h2+2*n/2,h2+3*n/2,n/2);
 doOtherOp<float><<<40,64,0,0>>>(f,f+n,f+2*n,f+3*n,n);

 cudaDeviceSynchronize();

}
