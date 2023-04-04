#pragma once

#include<cstdint>
#include<cmath>
#include<algorithm>

template<int NBITS=32> 
class FastPoissonPDF {
public:
  static_assert(NBITS<=32);
  constexpr uint64_t max = (1ULL<<NBITS) -1ULL;

  FastPoissonPDF(double mu) { reset(mu); }

  template<typename G>
  int operator()(G& g) { return find(g()); }

  void reset(double mu) {
    m_cumulative.clear();
    auto poiss = std::exp(-mu);
    double sum = poiss;
    double mul = max;
    m_cumulative.push_back(mul*sum+0.5);
    if (mu>0)
    for (;;) {
      poiss *= mu / float(cumulative.size());
      sum += poiss;
      if (uint64_t(mul*sum+0.5) >= max)
          break;
      cumulative.push_back(mul*sum+0.5);
    }
  }

  int find(uint32_t x) const {
      auto bin = std::lower_bound(m_cumulative.begin(), m_cumulative.end(), x);
      return bin - m_cumulative.begin();
  }



private:
  std::vector<uint32_t> m_cumulative;

};


