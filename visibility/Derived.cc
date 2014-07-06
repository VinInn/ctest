#include "Derived.h"
#include<iostream>
#include<typeinfo>

D::~D(){}

void D::hi() const {
  std::cout << "D" << std::endl;
  std::cout << "D " << typeid(*this).name() << std::endl;
  std::cout << "D " << &typeid(*this) << std::endl;
}
  
void D::who(Base const & b) const {
  std::cout << "who in D" << std::endl;
  std::cout << b.i1() << " " << b.i2()  << std::endl;
  std::cout << typeid(b).name() << " " << &typeid(b) << std::endl;
}
  
