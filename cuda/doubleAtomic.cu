#include<cassert>
#include<array>

struct DoubleCounter {

  using c_type = unsigned long long int;
 
  DoubleCounter(){}
  DoubleCounter(c_type i) { counter.ac=i;}
  DoubleCounter & operator=(c_type i) { counter.ac=i; return *this;}

  struct Counters { 
    uint32_t n;  // total size 
    uint32_t m;  // number of elements
  };

  union DoubleAtomic {
    Counters counters;
    c_type ac; //
  };

  static constexpr c_type incr = 1UL<<32;
  
  __device__ __host__
  Counters get() const { return counter.counters;}

  __device__
  Counters add(c_type i) {
    assert(i<incr);
    i+=incr;
    DoubleAtomic ret;
    ret.ac = atomicAdd(&counter.ac,i);
    return ret.counters;
  } 


  DoubleAtomic counter;
  
};


__global__
void update(DoubleCounter * dc,  uint32_t * ind, uint32_t * cont,  uint32_t n) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i>=n) return;

  auto m = i%11;
  m = m%6 +1;  // max 6, no 0
  auto c = dc->add(m);
  assert(c.m<n);
  ind[c.m] = c.n;
  for(int j=c.n; j<c.n+m; ++j) cont[j]=i; 

};

__global__
void finalize(DoubleCounter const * dc,  uint32_t * ind, uint32_t * cont,  uint32_t n) {
  assert(dc->get().m==n);
  ind[n]= dc->get().n;
}

__global__
void verify(DoubleCounter const * dc, uint32_t const * ind, uint32_t const * cont,  uint32_t n) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i>=n) return;
  assert(0==ind[0]);
  assert(dc->get().m==n);
  assert(ind[n] == dc->get().n);
  auto ib = ind[i];
  auto ie = ind[i+1];
  auto k = cont[ib++];
  assert(k<n);
  for (;ib<ie; ++ib) assert(cont[ib]==k);
}

#include<iostream>
int main() {

    DoubleCounter * dc_d;
    cudaMalloc(&dc_d, sizeof(DoubleCounter));
    cudaMemset(dc_d, 0, sizeof(DoubleCounter));

    printf("size %d\n",sizeof(DoubleCounter));

    constexpr uint32_t N=20000;
    constexpr uint32_t M=N*6;
    uint32_t *n_d, *m_d;
    cudaMalloc(&n_d, N*sizeof(int));
    // cudaMemset(n_d, 0, N*sizeof(int));
    cudaMalloc(&m_d, M*sizeof(int));


    update<<<2000, 512 >>>(dc_d,n_d,m_d,10000);
    finalize<<<2000, 512 >>>(dc_d,n_d,m_d,10000);
    verify<<<2000, 512 >>>(dc_d,n_d,m_d,10000);

    DoubleCounter dc;
    cudaMemcpy(&dc, dc_d, sizeof(DoubleCounter), cudaMemcpyDeviceToHost);

    std::cout << dc.get().n << ' ' << dc.get().m << std::endl;

    return 0;
}
