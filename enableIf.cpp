#include  <type_traits>
#include <mutex>

namespace a {
template<typename T>
inline constexpr bool isF =  std::is_floating_point<T>::value;

template<typename T>
inline constexpr bool isI =  std::is_integral<T>::value;
}

namespace b {

template<typename T, typename V
#ifdef SFINAE
         ,typename = std::enable_if_t<a::isF<T> and a::isI<V>>
#endif
       	>
struct QWRT  {
  friend class A;
#ifndef SFINAE
  static_assert(a::isF<T> and b::isI<V>);
#endif
  QWRT(int q) : f(q),i(q){}
  mutable std::mutex mutex_;
  private:
  T f;
  V i;
};

}

namespace c {

inline int get42() {
  static int * i = new int(42);
  return  *i;
}

template<typename V, typename = std::enable_if_t<a::isI<V>>>
inline auto & getIt() {
  static b::QWRT<float,V> qwrt(get42());
  return qwrt;
}
}

auto & get() {
   return c::getIt<int>();
}
