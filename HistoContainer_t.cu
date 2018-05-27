#include "HistoContainer.h"

#include<algorithm>
#include<cassert>
#include<iostream>
#include<random>
#include<limits>

#include "cuda/api_wrappers.h"


template<typename T>
void go() {

  if (cuda::device::count() == 0) {
	std::cerr << "No CUDA devices on this system" << "\n";
	exit(EXIT_FAILURE);
  }

  auto current_device = cuda::device::current::get(); 
 


  std::mt19937 eng;
  std::uniform_int_distribution<T> rgen(std::numeric_limits<T>::min(),std::numeric_limits<T>::max());

  
  constexpr int N=12000;
  T v[N];
  auto v_d = cuda::memory::device::make_unique<T[]>(current_device, N);

  cuda::memory::copy(v_d.get(), v, N*sizeof(T));


  using Hist = HistoContainer<T,7,8>;
  std::cout << "HistoContainer " << Hist::nbins << ' ' << Hist::binSize << std::endl;
  
  Hist h;

  auto h_d = cuda::memory::device::make_unique<Hist[]>(current_device, 1);

  for (int it=0; it<5; ++it) {
    for (long long j = 0; j < N; j++) v[j]=rgen(eng);

    cuda::memory::copy(v_d.get(), v, N*sizeof(T));

    fillFromVector(h_d.get(),v_d.get(),N,256,0);

    cuda::memory::copy(&h, h_d.get(), sizeof(Hist));                                

        
    std::cout << "nspills " << h.nspills << std::endl;    

    auto verify = [&](uint32_t i, uint32_t k, uint32_t t1, uint32_t t2) {
      assert(t1<N); assert(t2<N);
      if ( T(v[t1]-v[t2])<=0) std::cout << "for " << i <<':'<< v[k] <<" failed " << v[t1] << ' ' << v[t2] << std::endl;
    };

    for (uint32_t i=0; i<Hist::nbins; ++i) {
      if (0==h.n[i]) continue;
      auto k= *h.begin(i);
      assert(k<N);
      auto kl = h.bin(v[k]-T(1000));
      auto kh =	h.bin(v[k]+T(1000));
      assert(kl!=i);  assert(kh!=i);
      // std::cout << kl << ' ' << kh << std::endl;
      for (auto j=h.begin(kl); j<h.end(kl); ++j) verify(i,k,k,(*j));
      for (auto	j=h.begin(kh); j<h.end(kh); ++j) verify(i,k,(*j),k);
    }
  }

}

int main() {
  go<int16_t>();

  return 0;
}
