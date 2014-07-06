#include<string>
#include<iostream>

namespace cond {

  template<typename T>
  class Wrapper {
  public:
    typedef T Class;
    std::string print() const;

    std::string common() const {
      return "common";
    }



    Class * p;
  };


}


namespace mine {

  class A {
    int i;
  };


}

namespace yours {

  class B {
    float k;
  };


}


namespace cond {
  template<> 
  std::string Wrapper<mine::A>::print() const {
    return "mine::A";
  };

  template<> 
  std::string Wrapper<yours::B>::print() const {
    return "yours::B";
  };


}

int main() {

  cond::Wrapper<mine::A> wa;
  cond::Wrapper<yours::B> wb;

  std::cout << wa.print() << std::endl;
  std::cout << wb.print() << std::endl;

  std::cout << wa.common() << std::endl;
  std::cout << wb.common() << std::endl;

  return 0;
}
