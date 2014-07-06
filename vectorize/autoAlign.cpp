#include <iostream>


struct A {  int d[4]; };


struct alignas(16) B { int d[4];};

int main() {
  A a; B b; alignas(16) A c;
  std::cout << alignof(a) << ' ' << alignof(b) << ' ' << alignof(c) << std::endl;
  auto a1 =a; auto b1 = b; auto c1=c;
  std::cout << alignof(a1) << ' ' << alignof(b1) << ' ' << alignof(c1) << std::endl;

  A & aa =a; B & bb = b; A &  cc=c;
  std::cout << alignof(aa) << ' ' << alignof(bb) << ' ' << alignof(cc) << std::endl;
  auto & a2 =a; auto & b2 = b; auto & c2=c;
  std::cout << alignof(a2) << ' ' << alignof(b2) << ' ' << alignof(c2)<< std::endl;
  decltype(auto) a3 =a; decltype(auto)  b3 = b; decltype(auto)  c3=c;
  std::cout << alignof(a3) << ' ' << alignof(b3) << ' ' << alignof(c3)<< std::endl;


}
