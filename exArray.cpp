#include <array>
#include <initializer_list>


constexpr std::array<int, 4> a{1,2,3,4};

constexpr std::initializer_list<int>  al={1,2,3,4};


struct Base {
  explicit Base(std::initializer_list<int> me) {
    for ( auto a: me) k+=a;
  }

  int k=0;

};

struct Foo : public Base {

  template<typename... Args>
  Foo(int iq, Args... args) : Base({args...}), q(iq){}

    int q;
};


#include <iostream>

int main() {

  Foo f(1,1,2,3,4);
  std::cout << f.q << " " << f.k << std::endl;

  for (auto const & x : a) std::cout << x << " ";
  std::cout << std::endl;

  for (auto const & x : al) std::cout << x << " ";
  std::cout << std::endl;

  int c[2*al.size()];
  int i=0;
  for (auto const & x : al) {
    c[i++] = x;  c[i++] = x;
  }

  for (auto const & x : c) std::cout << x << " ";
  std::cout << std::endl;

  return 0;

} 
