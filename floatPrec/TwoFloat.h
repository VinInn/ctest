#pragma once
#include <cmath>
#include <algorithm>

namespace detailsTwoFloat {

// imported from https://gitlab.inria.fr/core-math/core-math/-/blob/master/src/binary64/pow/pow.h
// original version coded by H.  at CERN

/* Add a + b, such that hi + lo approximates a + b.
   Assumes |a| >= |b|.
   For rounding to nearest we have hi + lo = a + b exactly.
   For directed rounding, we have
   (a) hi + lo = a + b exactly when the exponent difference between a and b
       is at most 53 (the binary64 precision)
   (b) otherwise |(a+b)-(hi+lo)| <= 2^-105 min(|a+b|,|hi|)
       (see https://hal.inria.fr/hal-03798376)
   We also have |lo| < ulp(hi). */
template<typename T>
inline void fast_two_sum(T& hi, T& lo, T a, T b) {
  T e;

  // assert (a == 0 || std::abs (a) >= std::abs (b));
  hi = a + b;
  e = hi - a; /* exact */
  lo = b - e; /* exact */
}

/* Algorithm 2 from https://hal.science/hal-01351529 */
template<typename T>
inline void two_sum (T& s, T& t, T a, T b)
{
  s = a + b;
  T a_prime = s - b;
  T b_prime = s - a_prime;
  T delta_a = a - a_prime;
  T delta_b = b - b_prime;
  t = delta_a + delta_b;
}

// Add a + (bh + bl), assuming |a| >= |bh|
template<typename T>
inline void fast_sum(T& hi, T& lo, T a, T bh,
                            T bl) {
  fast_two_sum(hi, lo, a, bh);
  /* |(a+bh)-(hi+lo)| <= 2^-105 |hi| and |lo| < ulp(hi) */
  lo += bl;
  /* |(a+bh+bl)-(hi+lo)| <= 2^-105 |hi| + ulp(lo),
     where |lo| <= ulp(hi) + |bl|. */
}

// Multiply exactly a and b, such that hi + lo = a * b.
template<typename T>
inline void a_mul(T& hi, T& lo, T a, T b) {
  hi = a * b;
  lo = std::fma (a, b, -hi);
}

// Multiply a T with a T T : a * (bh + bl)
template<typename T>
inline void s_mul (T& hi, T& lo, T a, T bh,
                          T bl) {
  T s;

  a_mul (hi, s, a, bh); /* exact */
  lo = std::fma (a, bl, s);
  /* the error is bounded by ulp(lo), where |lo| < |a*bl| + ulp(hi) */
}

// Returns (ah + al) * (bh + bl) - (al * bl)
// We can ignore al * bl when assuming al <= ulp(ah) and bl <= ulp(bh)
template<typename T>
inline void d_mul(T&hi, T&lo, T ah, T al,
                         T bh, T bl) {
  T s, t;

  a_mul(hi, s, ah, bh);
  t = std::fma(al, bh, s);
  lo = std::fma(ah, bl, t);
}

template<typename T>
inline void d_square(T& hi, T& lo, T ah, T al) {
  T s, b = al + al;

  a_mul(hi, s, ah, ah);
  lo = std::fma(ah, b, s);
}

}


template<typename T>
class TwoFloat {
public:

  TwoFloat(){}
  explicit TwoFloat(T a) : mhi(a), mlo(0) {}

  TwoFloat(T a, T b) {
    using namespace detailsTwoFloat;
    fast_two_sum(mhi,mlo, std::max(a,b),std::min(a,b));
  }

  T hi() const { return mhi;}
  T lo() const { return mlo;}
  T & hi() { return mhi;}
  T & lo() { return mlo;}


  T mhi=0;
  T mlo=0;
};


template<typename T>
inline TwoFloat<T> operator+(TwoFloat<T> const & a, T b) {
  using namespace detailsTwoFloat;
  TwoFloat<T> ret(b,a.hi());
  auto u = ret.lo() + a.lo();
  fast_two_sum(ret.hi(), ret.lo(), ret.hi(),u);
  return ret;
}

template<typename T>
inline TwoFloat<T> operator+(T b, TwoFloat<T> const & a) {
  return a+b;
}

/* Algorithm 3 from https://hal.science/hal-01351529 */
template<typename T>
inline TwoFloat<T> operator*(TwoFloat<T> const & a, T b) {
  using namespace detailsTwoFloat;
  TwoFloat<T> ret;
  s_mul(ret.hi(),ret.lo(), b, a.hi(), a.lo());
  return ret;
}

template<typename T>
inline TwoFloat<T> operator*(T b, TwoFloat<T> const & a) {
  return a*b;
}

/* Algorithm 5 from https://hal.science/hal-01351529 */
template<typename T>
inline TwoFloat<T> operator+(TwoFloat<T> const & a, TwoFloat<T> const & b) {
  using namespace detailsTwoFloat;
  TwoFloat<T> ret(a.hi(), b.hi());
  auto u = a.lo() + b.lo();
  auto w = ret.lo() + u;
  fast_two_sum(ret.hi(), ret.lo(), ret.hi(),u);
  return ret;
}

/* Algorithm 11  from https://hal.science/hal-01351529 */
template<typename T>
inline TwoFloat<T> operator*(TwoFloat<T> const & a, TwoFloat<T> const & b) {
  using namespace detailsTwoFloat;
  TwoFloat<T> ret;
#ifdef MORE_PREC
  a_mul(ret.hi(),ret.lo(),a.hi(),b.hi());
  auto t0 =  a.lo() * b.lo();
  auto t1 =   std::fma(a.hi(),b.lo(),t0); 
  auto l2 =   std::fma(a.lo(),b.hi(),t1);
  auto l = ret.lo()+l2;
  fast_two_sum(ret.hi(), ret.lo(), ret.hi(),l);
#else
  d_mul(ret.hi(),ret.lo(),a.hi(),a.lo(),b.hi(),b.lo());
#endif
  return ret;
}
