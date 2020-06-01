
#include <cstdio>
#include <cassert>

#include <mma.h>
using namespace nvcuda;

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;


__constant__  float par0_32[16] = {2.,2.,2.,2., 2.,2.,2.,2., 2.,2.,2.,2., 2.,2.,2.,2.};
__device__  half par0[16];

__global__ void convertFp32ToFp16 (half *out, float const *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}

__global__ void wmma_example(half const * in, float * out, int size) {

   assert(size==16);
   if (threadIdx.x<size) { 
     par0[threadIdx.x] = par0_32[threadIdx.x];
     assert(2.f == par0_32[threadIdx.x]);
   }
   __syncthreads();

   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

   wmma::fill_fragment(acc_frag, 0.0f);
   

   wmma::load_matrix_sync(b_frag, par0, 4);


   wmma::load_matrix_sync(a_frag, in, 4);

   // Perform the matrix multiplication
   wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

   // Store the output
   wmma::store_matrix_sync(out, acc_frag, 4, wmma::mem_col_major);

}



__global__ void fill(half * in, int size) {

   if (threadIdx.x<size) {
     in[threadIdx.x] = float(threadIdx.x+1)*0.1f;
     assert(in[threadIdx.x]>half(0));
   }
   __syncthreads();
   if (threadIdx.x==0) printf("fill %f %f\n",float(in[0]),float(in[15]));

}


#include<iostream>

int main() {


   half * in_fp16;
   float * out_fp32;

   cudaMalloc(&in_fp16, 16 * sizeof(half));
   cudaMalloc(&out_fp32, 16 * sizeof(float));

   fill<<<1,32>>>(in_fp16,16);
   wmma_example<<<1,32>>>(in_fp16, out_fp32, 16);

   float loc[16];

   cudaMemcpy(loc,out_fp32, 16*sizeof(float), cudaMemcpyDeviceToHost);

   cudaDeviceSynchronize();

   for (auto f : loc) std::cout << f << ' ';
   std::cout << std::endl;

}
