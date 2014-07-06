#include "Base.h"
#include<iostream>
#include<typeinfo>

//#pragma visibility push(hidden) 
#pragma visibility push(internal)
namespace {

  class A : public Base {
  public:
    A(float a, float b) :
    Base (a, b) {}
    
    int i1() const { return 4;}

   virtual void who(Base const & b) const {
      std::cout << "who in A" << std::endl;
      std::cout << b.i1() << " " << b.i2()  << std::endl;
      std::cout << typeid(b).name() << " " << &typeid(b) << std::endl;
    }
    
    void hi() const {
      std::cout << "A" << std::endl;
      std::cout << "A " << typeid(*this).name() << std::endl;
      std::cout << "A " << &typeid(*this) << std::endl;
    }

  };

  struct FactoryA : public Factory<Base> {
    pointer operator()() {
      return pointer(new A(2.,3.14));
    } 
    
  };

  struct hello {
    hello() {
      std::cout << "\nhello A" << std::endl;
      A a (123,-123);
      a.who(a);
      a.hi();
      std::cout << std::endl;
    }
  };
  hello hi;

}
#pragma visibity pop


// extern "C" Factory<Base>* factoryA() __attribute__((vinPlug("A")));

extern "C" Factory<Base> * factoryA() {
  static FactoryA local;
  return &local;
}
