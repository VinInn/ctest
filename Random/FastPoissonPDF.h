#pragma once

#include "luxFloat.h"

#include<cstdint>
#include<cmath>
#include<algorithm>
#include<vector>


template<int NBITS=32> 
class FastPoissonPDF {
public:
  static_assert(NBITS<=32);
  constexpr static uint64_t max = (1ULL<<NBITS) -1ULL;

  using StorageType = typename BitStorage<1+(NBITS-1)/8>::type;
  static_assert(max<=std::numeric_limits<StorageType>::max());

  FastPoissonPDF(double mu) { reset(mu); }

  template<typename G>
  int operator()(G& g) {
    static_assert(G::max() == max); 
    return find(g()); 
  }

  void reset(double mu) {
    m_cumulative.clear();
    auto poiss = std::exp(-mu);
    double sum = poiss;
    double mul = max;
    m_cumulative.push_back(mul*sum+0.5);
    if (mu>0)
    for (;;) {
      poiss *= mu / m_cumulative.size();
      sum += poiss;
      if (uint64_t(mul*sum+0.5) >= max)
          break;
      m_cumulative.push_back(mul*sum+0.5);
    }
  }

  int find(uint32_t x) const {
      auto bin = std::lower_bound(m_cumulative.begin(), m_cumulative.end(), x);
      return bin - m_cumulative.begin();
  }


  auto const & cumulative() const { return m_cumulative; }

private:
  std::vector<StorageType> m_cumulative;

};


