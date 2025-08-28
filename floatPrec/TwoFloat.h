#pragma once

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
  *lo += bl;
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
pulic:

  T mhi;
  T mlo;
};

