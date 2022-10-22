#include  <type_traits>

template<typename T>
constexpr bool isF =  std::is_floating_point<T>::value;

template<typename T>
constexpr bool isI =  std::is_integral<T>::value;


template<typename T, typename V
#ifdef SFINAE
         ,typename = std::enable_if_t<isF<T> and isI<V>>
#endif
       	>
struct QWRT  {
#ifndef SFINAE
  static_assert(isF<T> and isI<V>);
#endif
  QWRT(int q) : f(q),i(q){}
  T f;
  V i;
};

template<typename T, typename V
        >
inline auto & getIt() {
  static QWRT<T,V> qwrt(42);
  return qwrt;

}


auto & get() {
   return getIt<float,float>();
}
