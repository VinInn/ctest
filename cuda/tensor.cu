#include <mma.h>
#include <cstdio>
using namespace nvcuda;

__global__ void wmma_ker(half *a, half *b, float *c) {
   // Declare the fragments
   constexpr int s = 16;
   wmma::fragment<wmma::matrix_a, s, s, s, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, s, s, s, half, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, s, s, s, float> c_frag;

   // Initialize the output to zero
   wmma::fill_fragment(c_frag, 0.0f);

   // Load the inputs
   wmma::load_matrix_sync(a_frag, a, 0);
   wmma::load_matrix_sync(b_frag, b, 0);
   __syncthreads();
if (threadIdx.x==0) {
   for(int i=0; i < a_frag.num_elements; i++) 
        printf("%f ",float(a_frag.x[i]));
   printf("\n%d\n",a_frag.num_elements);
}

   // Perform the matrix multiplication
   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

   // Store the output
   wmma::store_matrix_sync(c, c_frag, 0, wmma::mem_row_major);
}

#include <iostream>
int main() {

  half * a;
  half *  b;
  float * c;

  int n = 16;

  cudaMallocManaged(&a, n*sizeof(half));
  cudaMallocManaged(&b, n*sizeof(half));
  cudaMallocManaged(&c, n*sizeof(float));

  std::cout << sizeof(half) << ' ' << sizeof(float) << std::endl;


  for (int i=0; i<4; ++i) {
    c[i] = 0;
    a[i] = -1.;
    b[i] = 1.;
  }
  for (int i=4;  i<n; ++i) {
    c[i] = 0;
    a[i] = 1.;
    b[i] = 1.;
  }

  for (int i=0;  i<n; ++i) std::cout << float(a[i]) << ' ';
  std::cout << std::endl;
  for (int i=0;  i<n; ++i) std::cout << float(b[i]) << ' ';
  std::cout << std::endl;


  wmma_ker<<<1,16,0,0>>>(a,b,c);
  cudaDeviceSynchronize();


  for (int i=0;  i<n; ++i) std::cout << c[i] << ' ';
  std::cout << std::endl;

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  return 0;
}
