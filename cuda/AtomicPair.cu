#include<cassert>
#include<array>

struct AtomicPairCounter {

  using c_type = unsigned long long int;
 
  AtomicPairCounter(){}
  AtomicPairCounter(c_type i) { counter.ac=i;}
  AtomicPairCounter & operator=(c_type i) { counter.ac=i; return *this;}

  struct Counters { 
    uint32_t n;  // total size 
    uint32_t m;  // number of elements
  };

  union Atomic2 {
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
    Atomic2 ret;
    ret.ac = atomicAdd(&counter.ac,i);
    return ret.counters;
  } 


  Atomic2 counter;
  
};


__global__
void update(AtomicPairCounter * dc,  uint32_t * ind, uint32_t * cont,  uint32_t n) {
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
void finalize(AtomicPairCounter const * dc,  uint32_t * ind, uint32_t * cont,  uint32_t n) {
  assert(dc->get().m==n);
  ind[n]= dc->get().n;
}

__global__
void verify(AtomicPairCounter const * dc, uint32_t const * ind, uint32_t const * cont,  uint32_t n) {
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

    AtomicPairCounter * dc_d;
    cudaMalloc(&dc_d, sizeof(AtomicPairCounter));
    cudaMemset(dc_d, 0, sizeof(AtomicPairCounter));

    printf("size %d\n",sizeof(AtomicPairCounter));

    constexpr uint32_t N=20000;
    constexpr uint32_t M=N*6;
    uint32_t *n_d, *m_d;
    cudaMalloc(&n_d, N*sizeof(int));
    // cudaMemset(n_d, 0, N*sizeof(int));
    cudaMalloc(&m_d, M*sizeof(int));


    update<<<2000, 512 >>>(dc_d,n_d,m_d,10000);
    finalize<<<1,1 >>>(dc_d,n_d,m_d,10000);
    verify<<<2000, 512 >>>(dc_d,n_d,m_d,10000);

    AtomicPairCounter dc;
    uint32_t n;
    cudaMemcpy(&dc, dc_d, sizeof(AtomicPairCounter), cudaMemcpyDeviceToHost);
    cudaMemcpy(&n, dc_d, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::cout << dc.get().n << ' ' << dc.get().m << ' ' << n << std::endl;

    return 0;
}
