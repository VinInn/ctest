#pragma once
#include <cmath>
#include <concepts>
#include <type_traits>
#include <algorithm>
#include<cassert>
#include<ostream>

namespace detailsTwoFloat {

// imported from https://gitlab.inria.fr/core-math/core-math/-/blob/master/src/binary64/pow/pow.h
// original version coded by Tom Hubrecht at CERN in 2022

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
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr void fast_two_sum(T& hi, T& lo, T a, T b) {
  T e;

//  assert (a == 0 || std::abs (a) >= std::abs (b));
  hi = a + b;
  e = hi - a; /* exact */
  lo = b - e; /* exact */
}

/* Algorithm 2 from https://hal.science/hal-01351529 */
template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr void two_sum (T& s, T& t, T a, T b)
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
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr void fast_sum(T& hi, T& lo, T a, T bh,
                            T bl) {
  fast_two_sum(hi, lo, a, bh);
  /* |(a+bh)-(hi+lo)| <= 2^-105 |hi| and |lo| < ulp(hi) */
  lo += bl;
  /* |(a+bh+bl)-(hi+lo)| <= 2^-105 |hi| + ulp(lo),
     where |lo| <= ulp(hi) + |bl|. */
}

// Multiply exactly a and b, such that hi + lo = a * b.
template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr void a_mul(T& hi, T& lo, T a, T b) {
  hi = a * b;
  lo = std::fma (a, b, -hi);
}

// Multiply a T with a T T : a * (bh + bl)
template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr void s_mul (T& hi, T& lo, T a, T bh,
                          T bl) {
  T s;

  a_mul (hi, s, a, bh); /* exact */
  lo = std::fma (a, bl, s);
  /* the error is bounded by ulp(lo), where |lo| < |a*bl| + ulp(hi) */
}

// Returns (ah + al) * (bh + bl) - (al * bl)
// We can ignore al * bl when assuming al <= ulp(ah) and bl <= ulp(bh)
template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr void d_mul(T&hi, T&lo, T ah, T al,
                         T bh, T bl) {
  T s, t;

  a_mul(hi, s, ah, bh);
  t = std::fma(al, bh, s);
  lo = std::fma(ah, bl, t);
}

template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr void d_square(T& hi, T& lo, T ah, T al) {
  T s, b = al + al;

  a_mul(hi, s, ah, ah);
  lo = std::fma(ah, b, s);
}


template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr void a_div(T& hi, T& lo, T a, T b) {
  auto t = a/b;
  a_mul(hi,lo,t,b);
  auto d = a - hi;
  d = d - lo;
  lo = d/b;
  hi = t;
}

template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr void s_div(T& hi, T& lo, T ah, T al, T b) {
  auto t = ah/b;
  a_mul(hi,lo,t,b);
  auto d = ah - hi;
  d = d - lo;
  d = d + al;
  lo = d/b;
  hi = t;
}

   enum class From { members, fastsum, sum, prod, div };

   template<From from>
   struct Tag {
     constexpr Tag() = default;
     static constexpr From value() { return from;}
   };

   constexpr auto fromMembers()  { return  Tag<From::members>();}
   constexpr auto fromFastSum()  { return Tag<From::fastsum>();}
   constexpr auto fromSum()  { return Tag<From::sum>();}
   constexpr auto fromProd()  { return Tag<From::prod>();}
   constexpr auto fromDiv()  { return Tag<From::div>();}

}


template<typename T>
class TwoFloat {
public:
#ifdef __NVCC__
     __device__ __host__
#endif
  constexpr TwoFloat() = default;
#ifdef __NVCC__
     __device__ __host__
#endif
  /*explicit*/ constexpr TwoFloat(T a) : mhi(a), mlo(0) {}
#ifdef __NVCC__
     __device__ __host__
#endif
  constexpr TwoFloat & operator=(T a) { mhi=a; mlo=0; return *this;}
#ifdef __NVCC__
     __device__ __host__
#endif
  /*explicit*/ constexpr operator T() const { return mhi;}

  template<detailsTwoFloat::From f>
#ifdef __NVCC__
     __device__ __host__
#endif
  constexpr TwoFloat(T a, T b, detailsTwoFloat::Tag<f>) {
    using namespace detailsTwoFloat;
    using Tag = detailsTwoFloat::Tag<f>;
    if constexpr (Tag::value()==From::members) {
      mhi=a; mlo=b;
    } else if constexpr (Tag::value()==From::fastsum) {
      fast_two_sum(mhi,mlo,a,b);
    } else if constexpr (Tag::value()==From::sum) {
      two_sum(mhi,mlo,a,b);
    } else if constexpr (Tag::value()==From::prod) {
      a_mul(mhi,mlo,a,b);
    } else if constexpr (Tag::value()==From::div) {
      a_div(mhi,mlo,a,b);
    }
  }


#ifdef __NVCC__
     __device__ __host__
#endif
//  constexpr TwoFloat operator-() const {  TwoFloat<T> ret(-mhi, -mlo, detailsTwoFloat::fromMembers()); return ret;}
  constexpr TwoFloat operator-() const {  return TwoFloat<T>{-mhi, -mlo, detailsTwoFloat::fromMembers()};}


  constexpr TwoFloat & operator-=(TwoFloat<T> const & a);
  constexpr TwoFloat & operator+=(TwoFloat<T> const & a);
  constexpr TwoFloat & operator*=(TwoFloat<T> const & a);
  constexpr TwoFloat & operator/=(TwoFloat<T> const & a);

  constexpr TwoFloat & operator-=(T a);
  constexpr TwoFloat & operator+=(T a);
  constexpr TwoFloat & operator*=(T  a);
  constexpr TwoFloat & operator/=(T  a);


  constexpr T hi() const { return mhi;}
  constexpr T lo() const { return mlo;}
  constexpr T & hi() { return mhi;}
  constexpr T & lo() { return mlo;}

private:

  T mhi;
  T mlo;

};

template<typename T>
std::ostream& operator<<(std::ostream& os, TwoFloat<T> const & t)
{
    os << t.hi() << ',' << t.lo();
    return os;
}

template<typename T>
//, typename U, typename = typename std::enable_if_t<std::is_same_v<T, U>>>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr TwoFloat<T> operator+(TwoFloat<T> const & a, T b) {
  // static_assert( std::is_same<T, U>() );
  using namespace detailsTwoFloat;
  TwoFloat<T> ret(b,a.hi(),fromSum());
  auto u = ret.lo() + a.lo();
  fast_two_sum(ret.hi(), ret.lo(), ret.hi(),u);
  return ret;
}

template<typename T>
//, typename U, typename = typename std::enable_if_t<std::is_same_v<T, U>>>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr TwoFloat<T> operator+(T b, TwoFloat<T> const & a) {
  return a+b;
}

template<typename T>
// , typename U, typename = typename std::enable_if_t<std::is_same_v<T, U>>>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr TwoFloat<T> operator-(TwoFloat<T> const & a, T b) {
  // static_assert( std::is_same<T, U>() );
  using namespace detailsTwoFloat;
  TwoFloat<T> ret(-b,a.hi(),fromSum());
  auto u = ret.lo() + a.lo();
  fast_two_sum(ret.hi(), ret.lo(), ret.hi(),u);
  return ret;
}

template<typename T>
//, typename U, typename = typename std::enable_if_t<std::is_same_v<T, U>>>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr TwoFloat<T> operator-(T b, TwoFloat<T> const & a) {
  // static_assert( std::is_same<T, U>() );
  using namespace detailsTwoFloat;
  TwoFloat<T> ret(b,-a.hi(),fromSum());
  auto u = ret.lo() - a.lo();
  fast_two_sum(ret.hi(), ret.lo(), ret.hi(),u);
  return ret;
}



/* Algorithm 3 from https://hal.science/hal-01351529 */
template<typename T>
//, typename U, typename = typename std::enable_if_t<std::is_same_v<T, U>>>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr TwoFloat<T> operator*(TwoFloat<T> const & a, T b) {
  // static_assert( std::is_same<T, U>() );
  using namespace detailsTwoFloat;
  TwoFloat<T> ret;
  s_mul(ret.hi(),ret.lo(), b, a.hi(), a.lo());
  return ret;
}

template<typename T>
//, typename U, typename = typename std::enable_if_t<std::is_same_v<T, U>>>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr TwoFloat<T> operator*(T b, TwoFloat<T> const & a) {
  // static_assert( std::is_same<T, U>() );
  return a*b;
}

template<typename T>
inline constexpr T toSingle(T a) { return a;}

template<typename T>
inline constexpr double toDouble(T a) { return a;}


/*
template<>
inline constexpr __float128  toSingle<__float128>(__float128 a) { return a;}
*/


template<typename T>
inline constexpr T toSingle(TwoFloat<T> const & a) { return a.hi();}

template<typename T>
inline constexpr double toDouble(TwoFloat<T> const & a) { return double(a.hi())+double(a.lo());}



#ifdef TWOFLOAT_PRECISE_SUM
#warning "FP2_PREC ON"
// Algorithm 6 from https://hal.science/hal-01351529
template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr TwoFloat<T> operator+(TwoFloat<T> const & a, TwoFloat<T> const & b) {
  using namespace detailsTwoFloat;
  TwoFloat<T> ret(a.hi(), b.hi(),fromSum());
  TwoFloat<T> t(a.lo(), b.lo(),fromSum());
  auto u = ret.lo() + t.hi();
  fast_two_sum(ret.hi(), ret.lo(), ret.hi(),u);
  auto w = ret.lo() + t.lo();
  fast_two_sum(ret.hi(), ret.lo(), ret.hi(),w);
  return ret;
}
template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr TwoFloat<T> operator-(TwoFloat<T> const & a, TwoFloat<T> const & b) {
  using namespace detailsTwoFloat;
  TwoFloat<T> ret(a.hi(), -b.hi(),fromSum());
  TwoFloat<T> t(a.lo(), -b.lo(),fromSum());
  auto u = ret.lo() + t.hi();
  fast_two_sum(ret.hi(), ret.lo(), ret.hi(),u);
  auto w = ret.lo() + t.lo();
  fast_two_sum(ret.hi(), ret.lo(), ret.hi(),w);
  return ret;
}
#else
// Algorithm 5 from https://hal.science/hal-01351529
template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr TwoFloat<T> operator+(TwoFloat<T> const & a, TwoFloat<T> const & b) {
  using namespace detailsTwoFloat;
  TwoFloat<T> ret(a.hi(), b.hi(),fromSum());
  auto u = a.lo() + b.lo();
  auto w = ret.lo() + u;
#ifdef FP2_FAST
#warning "FP2_FAST ON"
  ret.lo() = w;
#else
  fast_two_sum(ret.hi(), ret.lo(), ret.hi(),w);
#endif
  return ret;
}
template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr TwoFloat<T> operator-(TwoFloat<T> const & a, TwoFloat<T> const & b) {
  using namespace detailsTwoFloat;
  TwoFloat<T> ret(a.hi(), -b.hi(),fromSum());
  auto u = a.lo() - b.lo();
  auto w = ret.lo() + u;
#ifdef FP2_FAST
  ret.lo() = w;
#else
  fast_two_sum(ret.hi(), ret.lo(), ret.hi(),w);
#endif
  return ret;
}
#endif


/* Algorithm 11  from https://hal.science/hal-01351529 */
template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr TwoFloat<T> operator*(TwoFloat<T> const & a, TwoFloat<T> const & b) {
  using namespace detailsTwoFloat;
  TwoFloat<T> ret;
#ifdef TWOFLOAT_PRECISE_MULT
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


/* Algorithm 15 from https://hal.science/hal-01351529 */
template<typename T>
//, typename U, typename = typename std::enable_if_t<std::is_same_v<T, U>>>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr TwoFloat<T> operator/(TwoFloat<T> const & a, T b) {
  // static_assert( std::is_same<T, U>() );
  using namespace detailsTwoFloat;
  TwoFloat<T> ret;
  s_div(ret.hi(),ret.lo(), a.hi(), a.lo(),b);
  return ret;
}

/* Algorithm 17  from https://hal.science/hal-01351529 */
template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr TwoFloat<T> operator/(TwoFloat<T> const & a, TwoFloat<T> const & b) {
  using namespace detailsTwoFloat;
#ifdef TWOFLOAT_PRECISE_DIV
  auto t = T(1.)/b.hi();
  auto rh = std::fma(-b.hi(),t,T(1.));
  auto rl= -b.lo()*t;
  TwoFloat<T> e(rh,rl,fromFastSum());
  auto d = t*e;
  auto m = t+d;
  return a*m;
#else
  auto t = a.hi()/b.hi();
  TwoFloat<T> ret = b*t;
  auto p = a.hi() - ret.hi();
  auto d = a.lo() - ret.lo();
  d = p + d;
  d = d/b.hi();
  fast_two_sum(ret.hi(), ret.lo(), t, d);
  return ret;
#endif
}

template<typename T>
//, typename U, typename = typename std::enable_if_t<std::is_same_v<T, U>>>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr TwoFloat<T> operator/(T a, TwoFloat<T> const & b) {
  // static_assert( std::is_same<T, U>() );
  using namespace detailsTwoFloat;
#ifdef TWOFLOAT_PRECISE_DIV
  auto t = T(1.)/b.hi();
  auto rh = std::fma(-b.hi(),t,T(1.));
  auto rl= -b.lo()*t;
  TwoFloat<T> e(rh,rl,fromFastSum());
  auto d = t*e;
  auto m = t+d;
  return a*m;
#else
  auto t = a/b.hi();
  TwoFloat<T> ret = b*t;
  auto p = a - ret.hi();
  auto d = p - ret.lo();;
  d = d/b.hi();
  fast_two_sum(ret.hi(), ret.lo(), t, d);
  return ret;
#endif
}


template<std::floating_point T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr T sqrt(T a) {
   static_assert(std::is_floating_point_v<T>);
   return std::sqrt(a);
}


//  Algorithm 6 from https://hal.science/hal-03482567
template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr TwoFloat<T> sqrt(TwoFloat<T> const & a) {
  using namespace detailsTwoFloat;
  auto s = std::sqrt(a.hi());
  auto r = std::fma(-s,s,a.hi());
  r = a.lo() + r;
  r = r/(T(2)*s);
#ifdef FP2_FAST
  return TwoFloat<T>(s,r,fromMembers());
#else
  return TwoFloat<T>(s,r,fromFastSum());
#endif
}

template<std::floating_point T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr T square(T a) {
   static_assert(std::is_floating_point_v<T>);
   return a*a;
}

template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr TwoFloat<T> square(TwoFloat<T> const & a) {
  using namespace detailsTwoFloat;
  TwoFloat<T> ret;
  d_square(ret.hi(),ret.lo(),a.hi(),a.lo());
  return ret;
}

template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr TwoFloat<T> square2(T a) {
using namespace detailsTwoFloat;
   return  {a,a,fromProd()};
}

/*
template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr TwoFloat<T> square2(TwoFloat<T> const & a) {
  return square(a);
}
*/

template<std::floating_point T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr T fabs(T a) {
   static_assert(std::is_floating_point_v<T>);
   return std::abs(a);
}

template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr TwoFloat<T> fabs(TwoFloat<T> const & a) {
  using namespace detailsTwoFloat;
  return {fabs(a.hi()),fabs(a.lo()),fromMembers()};
}



template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline
constexpr TwoFloat<T> & TwoFloat<T>::operator-=(TwoFloat<T> const & a) {
   *this = *this -a;
   return *this;
}

template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline
constexpr TwoFloat<T> & TwoFloat<T>::operator+=(TwoFloat<T> const & a) {
   *this = *this +a;
   return *this;
}

template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline
constexpr TwoFloat<T> & TwoFloat<T>::operator*=(TwoFloat<T> const & a) {
   *this = *this *a;
   return *this;
}

template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline
constexpr TwoFloat<T> & TwoFloat<T>::operator/=(TwoFloat<T> const & a) {
   *this = *this /a;
   return *this;
}




template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline
constexpr TwoFloat<T> & TwoFloat<T>::operator-=(T a) {
   *this = *this -a;
   return *this;
}

template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline
constexpr TwoFloat<T> & TwoFloat<T>::operator+=(T a) {
   *this = *this +a;
   return *this;
}

template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline
constexpr TwoFloat<T> & TwoFloat<T>::operator*=(T a) {
   *this = *this *a;
   return *this;
}

template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
inline
constexpr TwoFloat<T> & TwoFloat<T>::operator/=(T  a) {
   *this = *this /a;
   return *this;
}



//  Algorithm 10 from https://hal.science/hal-03482567
template<typename V>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr auto squaredNorm(V const  & v, int n) -> typename std::remove_cvref<decltype(v[0])>::type {
   using TT = typename std::remove_cvref<decltype(v[0])>::type;
   using namespace detailsTwoFloat;
   TT a0 = square(v[0]); 
   TT a1 = square(v[1]);
   TT  sum{a0.hi(),a1.hi(),fromSum()};
   auto s = a0.lo()+a1.lo();
   for (int  i=2; i<n; ++i) {
      TT const & a = square(v[i]);
      sum += a.hi();
      s += a.lo(); 
   }
   return sum + s;
} 


template<typename V>
#ifdef __NVCC__
     __device__ __host__
#endif
inline constexpr auto squaredNorm2(V const  & v, int n) -> TwoFloat<typename std::remove_cvref<decltype(v[0])>::type> {
   using T = typename std::remove_cvref<decltype(v[0])>::type;
   using TT = TwoFloat<typename std::remove_cvref<decltype(v[0])>::type>;
   static_assert(std::is_floating_point_v<T>);
   using namespace detailsTwoFloat;
   TT a0 = square2(v[0]);
   TT a1 = square2(v[1]);
   TT  sum{a0.hi(),a1.hi(),fromSum()};
   auto s = a0.lo()+a1.lo();
   for (int  i=2; i<n; ++i) {
      TT const & a = square2(v[i]);
      sum += a.hi();
      s += a.lo();
   }
   return sum + s;
}
