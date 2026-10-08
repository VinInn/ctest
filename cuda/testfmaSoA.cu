//
// /opt/nvidia/nsight-compute/2025.2.1/ncu --metrics sm__sass_thread_inst_executed_op_dfma_pred_on.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,,sm__sass_thread_inst_executed_op_hfma_pred_on.sum ./a.out
// https://godbolt.org/z/675bee119

#include <cuda_fp16.h>

template<typename T>
struct SoA {
   using Type = T;
   T * __restrict__ x;
   T * __restrict__ y;
   T * __restrict__ z;
};


template<typename T>
__global__ void dofma( SoA<T>  y, SoA<T> const x, int n) {
   int tid = blockDim.x * blockIdx.x + threadIdx.x;
     if (tid>=n) return;
     for (auto j=tid; j<n; j+=(gridDim.x * blockDim.x))  {
       if constexpr (std::is_same<half2, typename std::remove_cv<T>::type>::value) {
        auto v = __hfma2(y.x[j],T(1.e-12,1.e-12),x.x[j]);
        auto w = __hfma2(y.y[j],T(1.e-12,1.e-12),x.y[j]);
        auto u = __hfma2(y.z[j],T(1.e-12,1.e-12),x.z[j]);
        y.x[j] =  (__hfma2(w,u,v));
        y.y[j] =  (__hfma2(v,u,w));
        y.z[j] =  (__hfma2(v,w,u));
       }else {
        auto v = (y.x[j]*T(1.e-12))+x.x[j];
        auto w = (y.y[j]*T(1.e-12))+x.y[j];
        auto u = (y.z[j]*T(1.e-12))+x.z[j];
        y.x[j] =  (v+w*u);
        y.y[j] =  (w+v*u);
        y.z[j] =  (u+v*w);
        }
     }
}


template<typename T> 
void launch(T* v, int n) {
   SoA<T> x;
   SoA<T> y;
   x.x = v;
   x.y = v+n;
   x.z = v+2*n;
   y.x = v+3*n;
   y.y = v+4*n;
   y.z = v+5*n;
   dofma<T><<<40,64,0,0>>>(x,y,n);
}

int main() {
 
 double * d;
 float * f;
 half  * h;

 int n = 240*1024;

 cudaMallocManaged(&d, 6*n*sizeof(double));
 cudaMallocManaged(&f, 6*n*sizeof(float));
 cudaMallocManaged(&h, 6*n*sizeof(half));
 half2 * h2 = (half2*)(h);

 launch(d,n);
 launch(f,n);
 launch(h,n);
 launch(h2,n/2);

 cudaDeviceSynchronize();

}
