#include "cstdint"
// Type your code here, or load an example.
__global__ void square(int* array,  int64_t * t, int n) {
     uint64_t start, end;
     int tid = blockDim.x * blockIdx.x + threadIdx.x;
#ifdef CLOCK     
    auto s = clock64();
#else    
    // Record start time
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
#endif
     if (tid<n)
        array[tid] = array[tid] * array[tid];
    
    // Record end time 
#ifdef CLOCK
       *t = clock64() -s;
#else
   asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end));
   *t = start -end;
#endif 
}

