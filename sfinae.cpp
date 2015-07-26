struct V {
#ifdef A
  int a() const { return 1;}
#else
  int b() const { return 2;}
#endif
};


template<typename T>
auto v1(T const & a) -> decltype(a.a()) { return a.a();}
template<typename T>
auto v1(T const & a) -> decltype(a.b()) { return a.b();}


#include <iostream>
int main() {

V a;

std::cout << v1(a) << std::endl;

}
