#include "HistoContainer.h"

#include<algorithm>
#include<cassert>
#include<iostream>
#include<random>
#include<limits>


int main() {
  std::mt19937 eng;
  std::uniform_int_distribution<int16_t> rgen(std::numeric_limits<int16_t>::min(),std::numeric_limits<int16_t>::max());

  
  constexpr int N=12000;
  int16_t v[N];

  using Hist = HistoContainer<int16_t,7,8>;
  std::cout << "HistoContainer " << Hist::nbins << ' ' << Hist::binSize << std::endl;
  
  Hist h;
  for (int i=0; i<1; ++i) {
    for (long long j = 0; j < N; j++) v[j]=rgen(eng);
    h.zero();
    for (long long j = 0; j < N; j++) h.fill(v[j]);
    
    std::cout << "nspills " << h.nspills << std::endl;    
    
  }
  return 0;
}

