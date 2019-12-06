#include <type_traits>

template<typename T>
void foo(T * t) {}

template<typename F, typename ...Args>
void func(F f, Args... args) {
  f(args...);
}


 template<typename T> struct TD;



void go() {

  int k=4;
  int * p = &k;

  func(foo<std::remove_pointer<decltype(p)>::type>,p);

//   TD<std::remove_pointer<decltype(p)>::type> td;

}

