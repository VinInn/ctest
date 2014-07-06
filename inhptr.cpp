#include <iostream>
#include<typeinfo>
struct A{int i;};
struct B : public A{int i;};
struct C : public B{int i;};

struct A1{virtual ~A1(){}; int i;};
struct B1{virtual ~B1(){}; int i;};
struct C1 : public A1, public B1 {virtual ~C1(){}; int i;};

struct Fake{virtual ~Fake(){};};

int main() {

  {
    C c;
    B * b = &c;
    A * a = &c;
    void * p;
    p = &c;
    std::cout << p << std::endl;
    p = b;
    std::cout << p << std::endl;
    p = a;
    std::cout << p << std::endl;
    std::cout << std::endl;
  }
  {
    C1 c;
    B1 * b = &c;
    A1 * a = &c;
    void * p;
    p = &c;
    std::cout << p << std::endl;
    p = b;
    std::cout << p << std::endl;
    p = a;
    std::cout << p << std::endl;
    Fake * f = (Fake *)p;
    std::cout << typeid(*f).name()<< std::endl;
    std::cout << std::endl;
  }
  {
    C1 c;
    B1 * b = &c;
    A1 * a = &c;
    void * p;
    p = dynamic_cast<void *>(&c);
    std::cout << p << std::endl;
    p = dynamic_cast<void *>(b);
    std::cout << p << std::endl;
    p = dynamic_cast<void *>(a);
    std::cout << p << std::endl;
    std::cout << std::endl;
  }

}
