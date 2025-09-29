#pragma once
#include <cmath>
#include <concepts>
#include <type_traits>
#include <algorithm>
#include<cassert>
#include<ostream>
#if defined(__x86_64__)
#include <x86intrin.h>
#if !defined(__CUDA_ARCH__)
constexpr bool onX86 = true;
#else
constexpr bool onX86 = false;
#endif
#else
constexpr bool onX86 = false;
#endif


#ifdef __NVCC__
#define TWOFLOAT_INLINE __device__ __host__ inline constexpr 
#else
#define TWOFLOAT_INLINE inline constexpr
#endif


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
TWOFLOAT_INLINE void fast_two_sum(T& hi, T& lo, T a, T b) {
  T e;

//  assert (a == 0 || std::abs (a) >= std::abs (b));
  hi = a + b;
  e = hi - a; /* exact */
  lo = b - e; /* exact */
}

/* Algorithm 2 from https://hal.science/hal-01351529 */
template<typename T>
TWOFLOAT_INLINE void two_sum (T& s, T& t, T a, T b)
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
TWOFLOAT_INLINE void fast_sum(T& hi, T& lo, T a, T bh,
                            T bl) {
  fast_two_sum(hi, lo, a, bh);
  /* |(a+bh)-(hi+lo)| <= 2^-105 |hi| and |lo| < ulp(hi) */
  lo += bl;
  /* |(a+bh+bl)-(hi+lo)| <= 2^-105 |hi| + ulp(lo),
     where |lo| <= ulp(hi) + |bl|. */
}

// Multiply exactly a and b, such that hi + lo = a * b.
template<typename T>
TWOFLOAT_INLINE void a_mul(T& hi, T& lo, T a, T b) {
  hi = a * b;
  lo = std::fma (a, b, -hi);
}

// Multiply a T with a T T : a * (bh + bl)
template<typename T>
TWOFLOAT_INLINE void s_mul (T& hi, T& lo, T a, T bh,
                          T bl) {
  T s;

  a_mul (hi, s, a, bh); /* exact */
  lo = std::fma (a, bl, s);
  /* the error is bounded by ulp(lo), where |lo| < |a*bl| + ulp(hi) */
}

// Returns (ah + al) * (bh + bl) - (al * bl)
// We can ignore al * bl when assuming al <= ulp(ah) and bl <= ulp(bh)
template<typename T>
TWOFLOAT_INLINE void d_mul(T&hi, T&lo, T ah, T al,
                         T bh, T bl) {
  T s, t;

  a_mul(hi, s, ah, bh);
  t = std::fma(al, bh, s);
  lo = std::fma(ah, bl, t);
}

template<typename T>
TWOFLOAT_INLINE void d_square(T& hi, T& lo, T ah, T al) {
  T s, b = al + al;

  a_mul(hi, s, ah, ah);
  lo = std::fma(ah, b, s);
}


template<typename T>
TWOFLOAT_INLINE void a_div(T& hi, T& lo, T a, T b) {
  auto t = a/b;
  a_mul(hi,lo,t,b);
  auto d = a - hi;
  d = d - lo;
  lo = d/b;
  hi = t;
}

template<typename T>
TWOFLOAT_INLINE void s_div(T& hi, T& lo, T ah, T al, T b) {
  auto t = ah/b;
  a_mul(hi,lo,t,b);
  auto d = ah - hi;
  d = d - lo;
  d = d + al;
  lo = d/b;
  hi = t;
}

   enum class From { members, fastsum, sum, prod, div, fdouble };

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
   constexpr auto fromDouble()  { return Tag<From::fdouble>();}

}


template<typename T>
class TwoFloat {
public:

  TWOFLOAT_INLINE TwoFloat() = default;
  TWOFLOAT_INLINE /*explicit*/ TwoFloat(T a) : mhi(a), mlo(0) {}
  TWOFLOAT_INLINE TwoFloat & operator=(T a) { mhi=a; mlo=0; return *this;}
  TWOFLOAT_INLINE /*explicit*/ operator T() const { return mhi;}


  template<std::floating_point D, detailsTwoFloat::From f, 
           typename = typename std::enable_if_t<detailsTwoFloat::Tag<f>::value()==detailsTwoFloat::From::fdouble>>
  TWOFLOAT_INLINE TwoFloat(D a, detailsTwoFloat::Tag<f>) : mhi(a), mlo(a-mhi) {}

  template<detailsTwoFloat::From f>
  TWOFLOAT_INLINE TwoFloat(T a, T b, detailsTwoFloat::Tag<f>) {
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
    } // else static_assert(false,"Tag not valid");
  }


  TWOFLOAT_INLINE TwoFloat operator-() const {  return {-mhi, -mlo, detailsTwoFloat::fromMembers()};}


  TWOFLOAT_INLINE TwoFloat & operator-=(TwoFloat<T> const & a);
  TWOFLOAT_INLINE TwoFloat & operator+=(TwoFloat<T> const & a);
  TWOFLOAT_INLINE TwoFloat & operator*=(TwoFloat<T> const & a);
  TWOFLOAT_INLINE TwoFloat & operator/=(TwoFloat<T> const & a);

  TWOFLOAT_INLINE TwoFloat & operator-=(T a);
  TWOFLOAT_INLINE TwoFloat & operator+=(T a);
  TWOFLOAT_INLINE TwoFloat & operator*=(T  a);
  TWOFLOAT_INLINE TwoFloat & operator/=(T  a);


  TWOFLOAT_INLINE T hi() const { return mhi;}
  TWOFLOAT_INLINE T lo() const { return mlo;}
  TWOFLOAT_INLINE T & hi() { return mhi;}
  TWOFLOAT_INLINE T & lo() { return mlo;}

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
TWOFLOAT_INLINE TwoFloat<T> operator+(TwoFloat<T> const & a, T b) {
  // static_assert( std::is_same<T, U>() );
  using namespace detailsTwoFloat;
  TwoFloat<T> ret(b,a.hi(),fromSum());
  auto u = ret.lo() + a.lo();
  fast_two_sum(ret.hi(), ret.lo(), ret.hi(),u);
  return ret;
}

template<typename T>
//, typename U, typename = typename std::enable_if_t<std::is_same_v<T, U>>>
TWOFLOAT_INLINE TwoFloat<T> operator+(T b, TwoFloat<T> const & a) {
  return a+b;
}

template<typename T>
// , typename U, typename = typename std::enable_if_t<std::is_same_v<T, U>>>
TWOFLOAT_INLINE TwoFloat<T> operator-(TwoFloat<T> const & a, T b) {
  // static_assert( std::is_same<T, U>() );
  using namespace detailsTwoFloat;
  TwoFloat<T> ret(-b,a.hi(),fromSum());
  auto u = ret.lo() + a.lo();
  fast_two_sum(ret.hi(), ret.lo(), ret.hi(),u);
  return ret;
}

template<typename T>
//, typename U, typename = typename std::enable_if_t<std::is_same_v<T, U>>>
TWOFLOAT_INLINE TwoFloat<T> operator-(T b, TwoFloat<T> const & a) {
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
TWOFLOAT_INLINE TwoFloat<T> operator*(TwoFloat<T> const & a, T b) {
  // static_assert( std::is_same<T, U>() );
  using namespace detailsTwoFloat;
  TwoFloat<T> ret;
  s_mul(ret.hi(),ret.lo(), b, a.hi(), a.lo());
  return ret;
}

template<typename T>
//, typename U, typename = typename std::enable_if_t<std::is_same_v<T, U>>>
TWOFLOAT_INLINE TwoFloat<T> operator*(T b, TwoFloat<T> const & a) {
  // static_assert( std::is_same<T, U>() );
  return a*b;
}

template<typename T>
TWOFLOAT_INLINE T toSingle(T a) { return a;}

template<typename T>
TWOFLOAT_INLINE double toDouble(T a) { return a;}


/*
template<>
TWOFLOAT_INLINE __float128  toSingle<__float128>(__float128 a) { return a;}
*/


template<typename T>
TWOFLOAT_INLINE T toSingle(TwoFloat<T> const & a) { return a.hi();}

template<typename T>
TWOFLOAT_INLINE double toDouble(TwoFloat<T> const & a) { return double(a.hi())+double(a.lo());}



#ifdef TWOFLOAT_PRECISE_SUM
#warning "FP2_PREC ON"
// Algorithm 6 from https://hal.science/hal-01351529
template<typename T>
TWOFLOAT_INLINE TwoFloat<T> operator+(TwoFloat<T> const & a, TwoFloat<T> const & b) {
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
TWOFLOAT_INLINE TwoFloat<T> operator-(TwoFloat<T> const & a, TwoFloat<T> const & b) {
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
TWOFLOAT_INLINE TwoFloat<T> operator+(TwoFloat<T> const & a, TwoFloat<T> const & b) {
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
TWOFLOAT_INLINE TwoFloat<T> operator-(TwoFloat<T> const & a, TwoFloat<T> const & b) {
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
TWOFLOAT_INLINE TwoFloat<T> operator*(TwoFloat<T> const & a, TwoFloat<T> const & b) {
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
TWOFLOAT_INLINE TwoFloat<T> operator/(TwoFloat<T> const & a, T b) {
  // static_assert( std::is_same<T, U>() );
  using namespace detailsTwoFloat;
  TwoFloat<T> ret;
  s_div(ret.hi(),ret.lo(), a.hi(), a.lo(),b);
  return ret;
}

/* Algorithm 17  from https://hal.science/hal-01351529 */
template<typename T>
TWOFLOAT_INLINE TwoFloat<T> operator/(TwoFloat<T> const & a, TwoFloat<T> const & b) {
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
TWOFLOAT_INLINE TwoFloat<T> operator/(T a, TwoFloat<T> const & b) {
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
TWOFLOAT_INLINE T sqrt(T a) {
   static_assert(std::is_floating_point_v<T>);
   return std::sqrt(a);
}


//  Algorithm 6 from https://hal.science/hal-03482567
template<typename T>
TWOFLOAT_INLINE TwoFloat<T> sqrt(TwoFloat<T> const & a) {
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
TWOFLOAT_INLINE T square(T a) {
   static_assert(std::is_floating_point_v<T>);
   return a*a;
}

template<typename T>
TWOFLOAT_INLINE TwoFloat<T> square(TwoFloat<T> const & a) {
  using namespace detailsTwoFloat;
  TwoFloat<T> ret;
  d_square(ret.hi(),ret.lo(),a.hi(),a.lo());
  return ret;
}

template<typename T>
TWOFLOAT_INLINE TwoFloat<T> square2(T a) {
using namespace detailsTwoFloat;
   return  {a,a,fromProd()};
}

/*
template<typename T>
#ifdef __NVCC__
     __device__ __host__
#endif
TWOFLOAT_INLINE TwoFloat<T> square2(TwoFloat<T> const & a) {
  return square(a);
}
*/

template<std::floating_point T>
TWOFLOAT_INLINE T fabs(T a) {
   static_assert(std::is_floating_point_v<T>);
   return std::abs(a);
}

template<typename T>
TWOFLOAT_INLINE TwoFloat<T> fabs(TwoFloat<T> const & a) {
  using namespace detailsTwoFloat;
  return {fabs(a.hi()),fabs(a.lo()),fromMembers()};
}



template<typename T>
TWOFLOAT_INLINE TwoFloat<T> & TwoFloat<T>::operator-=(TwoFloat<T> const & a) {
   *this = *this -a;
   return *this;
}

template<typename T>
TWOFLOAT_INLINE TwoFloat<T> & TwoFloat<T>::operator+=(TwoFloat<T> const & a) {
   *this = *this +a;
   return *this;
}

template<typename T>
TWOFLOAT_INLINE TwoFloat<T> & TwoFloat<T>::operator*=(TwoFloat<T> const & a) {
   *this = *this *a;
   return *this;
}

template<typename T>
TWOFLOAT_INLINE TwoFloat<T> & TwoFloat<T>::operator/=(TwoFloat<T> const & a) {
   *this = *this /a;
   return *this;
}




template<typename T>
TWOFLOAT_INLINE TwoFloat<T> & TwoFloat<T>::operator-=(T a) {
   *this = *this -a;
   return *this;
}

template<typename T>
TWOFLOAT_INLINE TwoFloat<T> & TwoFloat<T>::operator+=(T a) {
   *this = *this +a;
   return *this;
}

template<typename T>
TWOFLOAT_INLINE TwoFloat<T> & TwoFloat<T>::operator*=(T a) {
   *this = *this *a;
   return *this;
}

template<typename T>
TWOFLOAT_INLINE TwoFloat<T> & TwoFloat<T>::operator/=(T  a) {
   *this = *this /a;
   return *this;
}


template<std::floating_point T>
TWOFLOAT_INLINE T rsqrt(T a) {
#ifdef __NVCC__
if constexpr (std::is_same_v<T,float>)
   return ::rsqrtf(a);
else
   return ::rsqrt(a);
#endif
   return T(1)/std::sqrt(a);
} 

template<typename T>
TWOFLOAT_INLINE TwoFloat<T> rsqrt(TwoFloat<T> const & a) {
  using namespace detailsTwoFloat;
  auto x = a.hi();
  float r;
  if constexpr (onX86 && std::is_same_v<T,float>) {
     _mm_store_ss( &r, _mm_rsqrt_ss( _mm_load_ss( &x ) ) );
     // standard one NR iteration
     r =  r * (1.5f - 0.5f * x * (r * r));
  } else { r = rsqrt(x);}
   float rx = r*x;
   auto drx = std::fma(r, x, -rx);
   float h = std::fma(r,rx,-1.0f) + r*drx;
   auto dr = (0.5f*r)*h;
   dr += (0.5f*r)*(r*r)*a.lo();
   return {r,-dr,fromSum()};
}


//  Algorithm 10 from https://hal.science/hal-03482567
template<typename V>
TWOFLOAT_INLINE auto squaredNorm(V const  & v, int n) -> typename std::remove_cvref<decltype(v[0])>::type {
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
TWOFLOAT_INLINE auto squaredNorm2(V const  & v, int n) -> TwoFloat<typename std::remove_cvref<decltype(v[0])>::type> {
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
