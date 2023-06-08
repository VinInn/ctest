#pragma once

#include "luxloat.h"

#include "approx_exp.h"

#include<cstdint>
#include<cmath>
#include<algorithm>
#include<vector>



/// standard Knuth Algorithm
template<typename G, typename F>
inline
int knuthPoissonFromExp(float expmu, G & gen, F & f) {
  auto cut = expmu;
  int ret = -1;
  float x = 1.f;
  do {
    x*= f(eng());
    ++ret;
  } while (x>cut);
  return ret;
}
template<typename G, typename F>
inline
int knuthPoisson(float mu, G & gen, F & f) {
  auto cut = unsafe_expf<6>(-mu);
  return knuthPoissonFromExp(cut,gen,f);
}


// CHLEP approximation
//
// Quick Poisson deviate algorithm:
//
// The principle:  For very large mu, a poisson distribution can be approximated
// by a gaussian: return the integer part of mu + .5 + g where g is a unit 
// normal.  However, this yelds a miserable approximation at values as
// "large" as 100.  The primary problem is that the poisson distribution is 
// supposed to have a skew of 1/mu**2, and the zero skew of the Guassian 
// leads to errors of order as big as 1/mu**2.
//
// We substitute for the gaussian a quadratic function of that gaussian random.
// The expression looks very nearly like mu + .5 - 1/6 + g + g**2/(6*mu).  
// The small positive quadratic term causes the resulting variate to have 
// a positive skew; the -1/6 constant term is there to correct for this bias 
// in the mean.  By adjusting these two and the linear term, we can match the
// first three moments to high accuracy in 1/mu.
//
// The sigma used is not precisely sqrt(mu) since a rounded-off Gaussian
// has a second moment which is slightly larger than that of the Gaussian.  
// To compensate, sig is multiplied by a factor which is slightly less than 1.
void  poissonNormalApproxCoeff(float mu, float & sig, float & a0, float& a1, float & a2) {
  // Compute the coefficients defining the quadratic transformation from a
  // Gaussian to a Poisson:

  // The multiplier corrects for fact that discretization of the form
  // [gaussian+.5] increases the second moment by a small amount.
  float  sig2 = mu * (0.9998654f - 0.08346f/mu);
  sig = std::sqrt(sig2);

  auto t = 1.0f/sig2;

  a2 = t*float(1./6.) + t*t*float(1./324.);
  a1 = std::sqrt (1.0f-2.0f*a2*a2*sig2);
  a0 = mu + 0.5f - sig2 * a2;

  // The formula will be a0 + a1*x + a2*x*x where x has sigma of sq.
  // The coeffeicients are chosen to match the first THREE moments of the
  // true Poisson distribution.

}

// G is a 64bits RNG, F is the normal transform 
template<typename G, typename F>
int poissonNormalApprox(float mu, G & gen, F& f) {
  float sig, a0,a1,a2;
  poissonNormalApproxCoeff(mu,sig,a0,a1,a2);
  auto g = f(gen()).first; // a waste;
  g *= sig;
  return (a2*g + a1)*g + a0;
}

class PoissonNormalApprox {
public:
  PoissonNormalApprox(float mu) {
    poissonNormalApproxCoeff(mu,sig,a0,a1,a2);
  }

  void reset(float mu) {
    poissonNormalApproxCoeff(mu,sig,a0,a1,a2);
  }

  template<typename G, typename F>
  std::pair<int,int> operator()(G & gen, F & f) {
    auto g = f(gen());
    g.first *= sig;
    g.second *= sig;
    return {(a2*g.first + a1)*g.first + a0, (a2*g.second + a1)*g.second + a0};
  }

private:
  float sig, a0,a1,a2;
};

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


