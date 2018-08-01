// -I/data/vin/cmssw/slc7_amd64_gcc700/external/cub/1.8.0-gnimlf2/include/
// -I/cvmfs/cms.cern.ch/slc7_amd64_gcc630/external/cub/1.8.0-gnimlf2/include


// kenneth.roche@pnl.gov ; k8r@uw.edu
// richford@uw.edu

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef USECUB
#include "cub/cub.cuh"
#endif

// profile cuda kernels
#define CUDA_PROFILING /*enable profiling */
#define CUDA_MAX_STREAMS 3

#ifdef CUDA_PROFILING

#define INIT_CUDA_PROFILER                                      \
    cudaEvent_t cuda_start_time[CUDA_MAX_STREAMS];              \
    cudaEvent_t cuda_stop_time[CUDA_MAX_STREAMS];               \
    float cuda_run_time[CUDA_MAX_STREAMS];                      \
    int cuda_iter;                                              \
    for(cuda_iter=0; cuda_iter<CUDA_MAX_STREAMS; cuda_iter++)   \
    {                                                           \
        cudaEventCreate(&cuda_start_time[cuda_iter]);           \
        cudaEventCreate(&cuda_stop_time[cuda_iter]);            \
    }

#define DESTROY_CUDA_PROFILER                                   \
    for(cuda_iter=0; cuda_iter<CUDA_MAX_STREAMS; cuda_iter++)   \
    {                                                           \
        cudaEventDestroy( cuda_start_time[cuda_iter] );         \
        cudaEventDestroy( cuda_stop_time[cuda_iter] );          \
    }

#define START_CUDA_TIMING(stream)                               \
    cudaEventRecord( cuda_start_time[stream], stream );

#define STOP_CUDA_TIMING(stream)                                \
    cudaEventRecord( cuda_stop_time[stream], stream );          \
    cudaEventSynchronize( cuda_stop_time[stream] );             \
    cudaEventElapsedTime( &cuda_run_time[stream], cuda_start_time[stream], cuda_stop_time[stream] );

#define GET_CUDA_TIMING(stream,time) time=cuda_run_time[stream];

#define GET_CUDA_BANDWIDTH(stream, bytes_read,  bytes_written, bandwidth) \
    bandwidth=1.0e-6*(bytes_read+bytes_written)/cuda_run_time[stream]; // cuda time in milliseconds, want result in GB but start w/ Byte total 

#else

#define INIT_CUDA_PROFILER
#define DESTROY_CUDA_PROFILER
#define START_CUDA_TIMING(stream)
#define STOP_CUDA_TIMING(stream)
#define GET_CUDA_TIMING(stream,time)
#define GET_CUDA_BANDWIDTH(stream, bytes_read,  bytes_written, bandwidth)

#endif

//                      +-------------------------------+
//----------------------| Reduce with Shuffling Kernels |----------------------
//                      +-------------------------------+

__device__ double atomicAddDouble(double* address, double val)
{
#if __CUDA_ARCH__ < 600
    unsigned long long int* address_as_ull = 
	    (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#else
 return atomicAdd(address,val);
#endif
}

__device__ inline double __shfl_down_double(double var, unsigned int srcLane, int width=32) {
    return __shfl_down_sync(0xffffffff,var, srcLane, width);
/*
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_down(a.x, srcLane, width);
    a.y = __shfl_down(a.y, srcLane, width);
    return *reinterpret_cast<double*>(&a);
*/
}

/* Reduce values within a warp
 * After execution, thread 0 has the total reduced value in it's variable
 */
__inline__ __device__ double warpReduceSum(double val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
        val += __shfl_down_double(val, offset);
    return val;
}

/* Reduce values within a block
 * First, reduce values within each warp, then the first thread of
 * each warp writes its partial sum to shared memory. Finally, 
 * after synchronizing, the first warp reads from shared memory and
 * reduces again.
 */
__inline__ __device__ double blockReduceSum(double val) {

    static __shared__ double shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);     // Each warp performs partial reduction

    if (lane==0) shared[wid]=val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

    return val;
}

/* Reduce across a complete grid.
 * Use a grid stride loop. The first pass generates and stores
 * partial reduction results. The second reduces the partial results
 * into a single total.
 */
__global__ void deviceReduceKernel(double *in, double* out, int N) {
    double sum = 0.0;
    //reduce multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
            i < N; 
            i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x==0)
        out[blockIdx.x]=sum;
}

void deviceReduce(double *in, double* out, int N) {
    int threads = 512;
    int blocks = min((N + threads - 1) / threads, 1024);

    deviceReduceKernel<<<blocks, threads>>>(in, out, N);
    deviceReduceKernel<<<1, 1024>>>(out, out, blocks);
}

__global__ void deviceReduceKernelVector2(double * in, double * out, int N) {
    double sum = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N / 2; i += blockDim.x * gridDim.x) {
        double2 val = reinterpret_cast<double2 *>(in)[ i ];
        sum += val.x + val.y;
    }

    int i = idx + N / 2 * 2;
    if (i < N)
        sum += in[ i ];

    sum = blockReduceSum(sum);
    if (threadIdx.x == 0)
        out[ blockIdx.x ] = sum;
}

void deviceReduceVector2(double * in, double * out, int N) {
    int threads = 512;
    int blocks = min((N / 2 + threads - 1) / threads, 1024);

    deviceReduceKernelVector2<<<blocks, threads>>>(in, out, N);
    deviceReduceKernel<<<1, 1024>>>(out, out, blocks);
}

__global__ void deviceReduceKernelVector4(double * in, double * out, int N) {
    double sum = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N / 4; i += blockDim.x * gridDim.x) {
        double4 val = reinterpret_cast<double4 *>(in)[ i ];
        sum += (val.x + val.y) + (val.z + val.w);
    }

    int i = idx + N / 4 * 4;
    if (i < N)
        sum += in[ i ];

    sum = blockReduceSum(sum);
    if (threadIdx.x == 0)
        out[ blockIdx.x ] = sum;
}

void deviceReduceVector4(double * in, double * out, int N) {
    int threads = 512;
    int blocks = min((N / 4 + threads - 1) / threads, 1024);

    deviceReduceKernelVector4<<<blocks, threads>>>(in, out, N);
    deviceReduceKernel<<<1, 1024>>>(out, out, blocks);
}

/* Reduce across a complete grid using Atomic operations.
 * Reduce across the warp using shuffle, then have the first thread
 * of each warp atomically update the reduced value.
 */
__global__ void deviceReduceWarpAtomicKernel(double * in, double * out, int N) {
    double sum = double(0.0);
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < N; 
        i += blockDim.x * gridDim.x) {
        sum += in[i];
    }

    sum = warpReduceSum(sum);

    if (threadIdx.x % warpSize == 0)
        atomicAddDouble(out, sum);
}

void deviceReduceWarpAtomic(double *in, double * out, int N) {
    int threads=256;
    int blocks=min((N+threads-1)/threads,2048);

    cudaMemsetAsync(out, 0, sizeof(double));
    deviceReduceWarpAtomicKernel<<<blocks,threads>>>(in,out,N); 
}

/* Reduce across a complete grid using Atomic operations.
 * Reduce across the warp using shuffle, then have the first thread
 * of each warp atomically update the reduced value.
 */
__global__ void deviceReduceWarpAtomicKernelVector2(double * in, double * out, int N) {
    double sum = double(0.0);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < N / 2; i += blockDim.x * gridDim.x) {
        double2 val = reinterpret_cast<double2 *>(in)[ i ];
        sum += val.x + val.y;
    }

    int i = idx + N / 2 * 2;
    if (i < N)
        sum += in[ i ];

    sum = warpReduceSum(sum);

    if (threadIdx.x % warpSize == 0)
        atomicAddDouble(out, sum);
}

void deviceReduceWarpAtomicVector2(double *in, double * out, int N) {
    int threads=256;
    int blocks=min((N/2+threads-1)/threads,2048);

    cudaMemsetAsync(out, 0, sizeof(double));
    deviceReduceWarpAtomicKernelVector2<<<blocks,threads>>>(in,out,N); 
}

/* Reduce across a complete grid using Atomic operations.
 * Reduce across the warp using shuffle, then have the first thread
 * of each warp atomically update the reduced value.
 */
__global__ void deviceReduceWarpAtomicKernelVector4(double * in, double * out, int N) {
    double sum = double(0.0);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < N / 4; i += blockDim.x * gridDim.x) {
        double4 val = reinterpret_cast<double4 *>(in)[ i ];
        sum += (val.x + val.y) + (val.z + val.w);
    }

    int i = idx + N / 4 * 4;
    if (i < N)
        sum += in[ i ];

    sum = warpReduceSum(sum);

    if (threadIdx.x % warpSize == 0)
        atomicAddDouble(out, sum);
}

void deviceReduceWarpAtomicVector4(double *in, double * out, int N) {
    int threads=256;
    int blocks=min((N/4+threads-1)/threads,2048);

    cudaMemsetAsync(out, 0, sizeof(double));
    deviceReduceWarpAtomicKernelVector4<<<blocks,threads>>>(in,out,N); 
}

//                      +-------------------------------+
//----------------------| End Adam's Reduction Kernels  |----------------------
//                      +-------------------------------+

int opt_threads(int new_blocks, int threads, int current_size)
{
    int new_threads;

    if ( new_blocks == 1 ) {
        new_threads = 2; 
        while ( new_threads < threads ) { 
            if ( new_threads >= current_size ) 
                break;
            new_threads *= 2 ;
        }
    }
    else 
        new_threads = threads ;

    return new_threads;
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile double *sdata, unsigned int tid) 
{// extended to block size 64 for warp case --factors of two here 

    if (blockSize >= 64) sdata[tid] += sdata[tid + 32]; //from 64 to 32 
    if ( tid < 16 ) sdata[tid] += sdata[tid + 16]; // from 32 to 16
    if ( tid < 8 ) sdata[tid] += sdata[tid +  8]; // from 16 to 8
    if ( tid < 4 ) sdata[tid] += sdata[tid +  4]; // from 8 to 4
    if ( tid < 2 ) sdata[tid] += sdata[tid +  2]; // from 4 to 2 
    if ( tid == 0 ) sdata[tid] += sdata[tid +  1]; // from 2 to 1 
    // ... finished 

    /*
       if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
       if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
       if (blockSize >= 16) sdata[tid] += sdata[tid +  8];
       if (blockSize >=  8) sdata[tid] += sdata[tid +  4];
       if (blockSize >=  4) sdata[tid] += sdata[tid +  2];
       if (blockSize >=  2) sdata[tid] += sdata[tid +  1];
       */
}

template <unsigned int blockSize>
__global__ void __reduce_kernel__(double *g_idata, double *g_odata, int n)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int loff=i+blockDim.x;

    if ( loff < n ) 
        sdata[tid] = g_idata[i] + g_idata[loff]; //let these threads load more than a single element
    else if ( i < n ) 
        sdata[tid] = g_idata[i];
    else         
        sdata[tid] = (double)(0.0);

    __syncthreads();

    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >=  512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >=  256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >=  128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32) 
        warpReduce<blockSize> (sdata, tid) ;

    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}

void call_reduction_kernel(int blocks, int threads, int size, double *d_idata, double *d_odata)
{
    //1st call:  call_reduction_kernel(blocks, lthreads, size, array, partial_sums);

    int smemSize = threads * sizeof(double);
    switch ( threads )
    {
        case 1024:
            __reduce_kernel__<1024><<< blocks, threads, smemSize >>>(d_idata, d_odata, size); break;
        case 512:
            __reduce_kernel__< 512><<< blocks, threads, smemSize >>>(d_idata, d_odata, size); break;
        case 256:
            __reduce_kernel__< 256><<< blocks, threads, smemSize >>>(d_idata, d_odata, size); break;
        case 128:
            __reduce_kernel__< 128><<< blocks, threads, smemSize >>>(d_idata, d_odata, size); break;
        case 64:
            __reduce_kernel__<  64><<< blocks, threads, smemSize >>>(d_idata, d_odata, size); break;
    }   
}

/*
   function does fast reduction (sum of elements) of array. 
   result is located in partial_sums[0].
   IF partial_sums == array then array contents will be destroyed
   */ 
int local_reduction( double * array , int size , double * partial_sums , int blocks , int threads )
{
    //call: local_reduction(gpu_workbuf, nxyz, gpu_workbuf, gpu_blocks, gpu_threads); // routine goes back and forth between the host and device
    unsigned int new_blocks, current_size;
    unsigned int lthreads = threads / 2 ; // threads should be power of 2

    if ( lthreads < 64 ) 
        lthreads = 64 ; //at least 2*warp_size

    // First reduction of the array
    call_reduction_kernel(blocks, lthreads, size, array, partial_sums);

    // Do iteratively reduction of partial_sums
    current_size = blocks;
    while ( current_size > 1 )
    {
        new_blocks = (int)ceil((float)current_size/threads);
        lthreads = opt_threads( new_blocks , threads , current_size ) / 2 ;
        if ( lthreads < 64 ) 
            lthreads=64; // at least 2*warp_size
        call_reduction_kernel( new_blocks , lthreads , current_size , partial_sums , partial_sums ) ;
        current_size = new_blocks ;
    }    
    return 0;
}

int main ( int argc , char ** argv ) 
{
    int nxyz=256*256*256;
    cudaError err;

    if (argc==2) 
        nxyz = atoi(argv[1]);

    int gpu_threads=512;
    int gpu_blocks=(int)ceil((float)nxyz/gpu_threads);


    printf("GPU SETTING: THREADS=%d, BLOCKS=%d, THREADS*BLOCKS=%d, nxyz=%d\n",gpu_threads,gpu_blocks,gpu_threads*gpu_blocks,nxyz);

    // Buffers
    double * workbuf;
    double * gpu_workbuf;

    err = cudaHostAlloc( (void **)&workbuf , nxyz*sizeof(double), cudaHostAllocDefault );
    if ( err!= cudaSuccess ) { printf("ERROR: (line %d in %s)\n", __LINE__ , __FILE__ ) ; return 1; }

    err = cudaMalloc( (void **)&gpu_workbuf , nxyz*sizeof(double) );
    if ( err!= cudaSuccess ) { printf("ERROR: (line %d in %s)\n", __LINE__ , __FILE__ ) ; return 1; }

    // Fill buffer with determnistic numbers
    double step = 1. ;
    int i ;
    for ( i=0; i<nxyz; i++ ) 
        workbuf[i]=step*(i+1); // no cast here 

    err = cudaMemcpy( gpu_workbuf , workbuf , nxyz*sizeof(double) , cudaMemcpyHostToDevice );
    if ( err != cudaSuccess ) { printf("ERROR: (line %d in %s)\n", __LINE__, __FILE__); return 1; }

    // gpu reduction
    double gpu_redu ;
    double *gpu_reduction_tmp ;
    err = cudaMalloc( (void **)&gpu_reduction_tmp , gpu_blocks*sizeof(double) );
    if ( err != cudaSuccess ) { printf("ERROR: (line %d in %s)\n", __LINE__, __FILE__); return 1; }

    INIT_CUDA_PROFILER;

    // do reduction
    START_CUDA_TIMING(0);
    //local_reduction(gpu_workbuf, nxyz, gpu_workbuf, gpu_blocks, gpu_threads); // routine goes back and forth between the host and device
    local_reduction(gpu_workbuf, nxyz, gpu_reduction_tmp, gpu_blocks, gpu_threads); // routine goes back and forth between the host and device
    STOP_CUDA_TIMING(0);

    // gather timing results 
    double cuda_time , cuda_bandwidth ;
    GET_CUDA_TIMING( 0 , cuda_time ) ;
    GET_CUDA_BANDWIDTH( 0 , nxyz*sizeof(double) ,  gpu_blocks*sizeof(double) , cuda_bandwidth ) ;
    printf("\n***Original reduce6 routines ***\n");
    printf("CUDA: TIME=%fms, BANDWIDTH=%fGB/s\n", cuda_time, cuda_bandwidth ) ;

    // copy result and check it
    //err=cudaMemcpy( &gpu_redu , gpu_workbuf , sizeof(double) , cudaMemcpyDeviceToHost );
    err=cudaMemcpy( &gpu_redu , gpu_reduction_tmp , sizeof(double) , cudaMemcpyDeviceToHost );
    if(err!= cudaSuccess) { printf("ERROR: (line %d in %s), err=%d\n", __LINE__, __FILE__, err); return 1; }  

    printf("THEORY:\t %.8f\n",0.5*nxyz*(workbuf[0]+workbuf[nxyz-1]));
    printf("GPU:   \t %.8f\n",gpu_redu);

    // Do it all over again for the reduce with shuffle
    printf("\n***With Shuffling ***\n");
    printf("Now testing reduction with warp shuffle\n");

    err = cudaMemcpy( gpu_workbuf , workbuf , nxyz*sizeof(double) , cudaMemcpyHostToDevice );
    if ( err != cudaSuccess ) { printf("ERROR: (line %d in %s)\n", __LINE__, __FILE__); return 1; }

    // do reduction
    START_CUDA_TIMING(0);
    deviceReduce(gpu_workbuf, gpu_reduction_tmp, nxyz);
    STOP_CUDA_TIMING(0);

    int threads = 512;
    int blocks = min((nxyz + threads - 1) / threads, 1024);
    // gather timing results 
    GET_CUDA_TIMING( 0 , cuda_time ) ;
    GET_CUDA_BANDWIDTH( 0 , nxyz*sizeof(double) ,  gpu_blocks*sizeof(double) , cuda_bandwidth ) ;
    printf("CUDA: TIME=%fms, BANDWIDTH=%fGB/s\n", cuda_time, cuda_bandwidth ) ;

    // copy result and check it
    //err=cudaMemcpy( &gpu_redu , gpu_workbuf , sizeof(double) , cudaMemcpyDeviceToHost );
    err=cudaMemcpy( &gpu_redu , gpu_reduction_tmp , sizeof(double) , cudaMemcpyDeviceToHost );
    if(err!= cudaSuccess) { printf("ERROR: (line %d in %s), err=%d\n", __LINE__, __FILE__, err); return 1; }  

    printf("THEORY:\t %.8f\n",0.5*nxyz*(workbuf[0]+workbuf[nxyz-1]));
    printf("GPU:   \t %.8f\n",gpu_redu);

    // Do it all over again for the reduce with shuffle and vectorization
    printf("Now testing reduction with warp shuffle and double2 vectorization\n");

    err = cudaMemcpy( gpu_workbuf , workbuf , nxyz*sizeof(double) , cudaMemcpyHostToDevice );
    if ( err != cudaSuccess ) { printf("ERROR: (line %d in %s)\n", __LINE__, __FILE__); return 1; }

    // do reduction
    START_CUDA_TIMING(0);
    deviceReduceVector2(gpu_workbuf, gpu_reduction_tmp, nxyz);
    STOP_CUDA_TIMING(0);

    threads = 512;
    blocks = min((nxyz/2 + threads - 1) / threads, 1024);
    // gather timing results 
    GET_CUDA_TIMING( 0 , cuda_time ) ;
    GET_CUDA_BANDWIDTH( 0 , nxyz*sizeof(double) ,  gpu_blocks*sizeof(double) , cuda_bandwidth ) ;
    printf("CUDA: TIME=%fms, BANDWIDTH=%fGB/s\n", cuda_time, cuda_bandwidth ) ;

    // copy result and check it
    //err=cudaMemcpy( &gpu_redu , gpu_workbuf , sizeof(double) , cudaMemcpyDeviceToHost );
    err=cudaMemcpy( &gpu_redu , gpu_reduction_tmp , sizeof(double) , cudaMemcpyDeviceToHost );
    if(err!= cudaSuccess) { printf("ERROR: (line %d in %s), err=%d\n", __LINE__, __FILE__, err); return 1; }  

    printf("THEORY:\t %.8f\n",0.5*nxyz*(workbuf[0]+workbuf[nxyz-1]));
    printf("GPU:   \t %.8f\n",gpu_redu);

    // Do it all over again for the reduce with shuffle and vectorization
    printf("Now testing reduction with warp shuffle and double4 vectorization\n");

    err = cudaMemcpy( gpu_workbuf , workbuf , nxyz*sizeof(double) , cudaMemcpyHostToDevice );
    if ( err != cudaSuccess ) { printf("ERROR: (line %d in %s)\n", __LINE__, __FILE__); return 1; }

    // do reduction
    START_CUDA_TIMING(0);
    deviceReduceVector4(gpu_workbuf, gpu_reduction_tmp, nxyz);
    STOP_CUDA_TIMING(0);

    threads = 512;
    blocks = min((nxyz/4 + threads - 1) / threads, 1024);
    // gather timing results 
    GET_CUDA_TIMING( 0 , cuda_time ) ;
    GET_CUDA_BANDWIDTH( 0 , nxyz*sizeof(double) ,  gpu_blocks*sizeof(double) , cuda_bandwidth ) ;
    printf("CUDA: TIME=%fms, BANDWIDTH=%fGB/s\n", cuda_time, cuda_bandwidth ) ;

    // copy result and check it
    //err=cudaMemcpy( &gpu_redu , gpu_workbuf , sizeof(double) , cudaMemcpyDeviceToHost );
    err=cudaMemcpy( &gpu_redu , gpu_reduction_tmp , sizeof(double) , cudaMemcpyDeviceToHost );
    if(err!= cudaSuccess) { printf("ERROR: (line %d in %s), err=%d\n", __LINE__, __FILE__, err); return 1; }  

    printf("THEORY:\t %.8f\n",0.5*nxyz*(workbuf[0]+workbuf[nxyz-1]));
    printf("GPU:   \t %.8f\n",gpu_redu);

    // Do it all over again for the reduce with shuffle and atomics
    printf("\n*** With Atomics ***\n");
    printf("Now testing reduction with warp shuffle and atomics\n");

    err = cudaMemcpy( gpu_workbuf , workbuf , nxyz*sizeof(double) , cudaMemcpyHostToDevice );
    if ( err != cudaSuccess ) { printf("ERROR: (line %d in %s)\n", __LINE__, __FILE__); return 1; }

    // do reduction
    START_CUDA_TIMING(0);
    deviceReduceWarpAtomic(gpu_workbuf, gpu_reduction_tmp, nxyz);
    STOP_CUDA_TIMING(0);

    threads = 512;
    blocks = min((nxyz/4 + threads - 1) / threads, 1024);
    // gather timing results 
    GET_CUDA_TIMING( 0 , cuda_time ) ;
    GET_CUDA_BANDWIDTH( 0 , nxyz*sizeof(double) ,  gpu_blocks*sizeof(double) , cuda_bandwidth ) ;
    printf("CUDA: TIME=%fms, BANDWIDTH=%fGB/s\n", cuda_time, cuda_bandwidth ) ;

    // copy result and check it
    //err=cudaMemcpy( &gpu_redu , gpu_workbuf , sizeof(double) , cudaMemcpyDeviceToHost );
    err=cudaMemcpy( &gpu_redu , gpu_reduction_tmp , sizeof(double) , cudaMemcpyDeviceToHost );
    if(err!= cudaSuccess) { printf("ERROR: (line %d in %s), err=%d\n", __LINE__, __FILE__, err); return 1; }  

    printf("THEORY:\t %.8f\n",0.5*nxyz*(workbuf[0]+workbuf[nxyz-1]));
    printf("GPU:   \t %.8f\n",gpu_redu);

    // Do it all over again for the reduce with shuffle and atomics and vector2
    printf("Now testing reduction with warp shuffle and atomics and double2\n");

    err = cudaMemcpy( gpu_workbuf , workbuf , nxyz*sizeof(double) , cudaMemcpyHostToDevice );
    if ( err != cudaSuccess ) { printf("ERROR: (line %d in %s)\n", __LINE__, __FILE__); return 1; }

    // do reduction
    START_CUDA_TIMING(0);
    deviceReduceWarpAtomicVector2(gpu_workbuf, gpu_reduction_tmp, nxyz);
    STOP_CUDA_TIMING(0);

    threads = 512;
    blocks = min((nxyz/4 + threads - 1) / threads, 1024);
    // gather timing results 
    GET_CUDA_TIMING( 0 , cuda_time ) ;
    GET_CUDA_BANDWIDTH( 0 , nxyz*sizeof(double) ,  gpu_blocks*sizeof(double) , cuda_bandwidth ) ;
    printf("CUDA: TIME=%fms, BANDWIDTH=%fGB/s\n", cuda_time, cuda_bandwidth ) ;

    // copy result and check it
    //err=cudaMemcpy( &gpu_redu , gpu_workbuf , sizeof(double) , cudaMemcpyDeviceToHost );
    err=cudaMemcpy( &gpu_redu , gpu_reduction_tmp , sizeof(double) , cudaMemcpyDeviceToHost );
    if(err!= cudaSuccess) { printf("ERROR: (line %d in %s), err=%d\n", __LINE__, __FILE__, err); return 1; }  

    printf("THEORY:\t %.8f\n",0.5*nxyz*(workbuf[0]+workbuf[nxyz-1]));
    printf("GPU:   \t %.8f\n",gpu_redu);

    // Do it all over again for the reduce with shuffle and atomics and vector4
    printf("Now testing reduction with warp shuffle and atomics and double4\n");

    err = cudaMemcpy( gpu_workbuf , workbuf , nxyz*sizeof(double) , cudaMemcpyHostToDevice );
    if ( err != cudaSuccess ) { printf("ERROR: (line %d in %s)\n", __LINE__, __FILE__); return 1; }

    // do reduction
    START_CUDA_TIMING(0);
    deviceReduceWarpAtomicVector4(gpu_workbuf, gpu_reduction_tmp, nxyz);
    STOP_CUDA_TIMING(0);

    threads = 512;
    blocks = min((nxyz/4 + threads - 1) / threads, 1024);
    // gather timing results 
    GET_CUDA_TIMING( 0 , cuda_time ) ;
    GET_CUDA_BANDWIDTH( 0 , nxyz*sizeof(double) ,  gpu_blocks*sizeof(double) , cuda_bandwidth ) ;
    printf("CUDA: TIME=%fms, BANDWIDTH=%fGB/s\n", cuda_time, cuda_bandwidth ) ;

    // copy result and check it
    //err=cudaMemcpy( &gpu_redu , gpu_workbuf , sizeof(double) , cudaMemcpyDeviceToHost );
    err=cudaMemcpy( &gpu_redu , gpu_reduction_tmp , sizeof(double) , cudaMemcpyDeviceToHost );
    if(err!= cudaSuccess) { printf("ERROR: (line %d in %s), err=%d\n", __LINE__, __FILE__, err); return 1; }  

    printf("THEORY:\t %.8f\n",0.5*nxyz*(workbuf[0]+workbuf[nxyz-1]));
    printf("GPU:   \t %.8f\n",gpu_redu);


#ifdef USECUB
    printf("\n*** Now with CUB ***\n");
    size_t temp_storage_bytes;
    double * temp_storage = NULL;
    cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, gpu_workbuf, gpu_reduction_tmp, nxyz, cub::Sum(),0);
    cudaMalloc(&temp_storage, temp_storage_bytes);

    cudaDeviceSynchronize();
    START_CUDA_TIMING(0);
    cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, gpu_workbuf, gpu_reduction_tmp, nxyz, cub::Sum(),0);
    STOP_CUDA_TIMING(0);

    threads = 512;
    blocks = min((nxyz/4 + threads - 1) / threads, 1024);
    // gather timing results 
    GET_CUDA_TIMING( 0 , cuda_time ) ;
    GET_CUDA_BANDWIDTH( 0 , nxyz*sizeof(double) ,  gpu_blocks*sizeof(double) , cuda_bandwidth ) ;
    printf("CUDA: TIME=%fms, BANDWIDTH=%fGB/s\n", cuda_time, cuda_bandwidth ) ;

    // copy result and check it
    //err=cudaMemcpy( &gpu_redu , gpu_workbuf , sizeof(double) , cudaMemcpyDeviceToHost );
    err=cudaMemcpy( &gpu_redu , gpu_reduction_tmp , sizeof(double) , cudaMemcpyDeviceToHost );
    if(err!= cudaSuccess) { printf("ERROR: (line %d in %s), err=%d\n", __LINE__, __FILE__, err); return 1; }  

    printf("THEORY:\t %.8f\n",0.5*nxyz*(workbuf[0]+workbuf[nxyz-1]));
    printf("GPU:   \t %.8f\n",gpu_redu);
#endif



    DESTROY_CUDA_PROFILER ;

    return 0 ;
}

