#define protected public
#include<typeinfo>
#include<iostream>


struct A {
  virtual ~A(){}


};


struct B : public A { int k;};

struct C{};


template<typename T> 
void print() {
  std::cout << sizeof(T) << std::endl;
  std::type_info const & t = typeid(T);
  std::cout << t.name() << std::endl;
  std::cout << t.__name << std::endl;
  std::cout << t.hash_code() << std::endl;
}


template<typename T>
void print(T const & p) {
  std::cout << sizeof(p) << std::endl;
  std::type_info const & t = typeid(p);
  std::cout << t.name() << std::endl;
  std::cout << t.__name << std::endl;
  std::cout << t.hash_code() << std::endl;
}



int main() {
  print<A>();
  print<B>();
  print<C>();

  A a;
  B b;
  C c;

  print(a);
  print(b);
  print(c);
  A * p = &b;
  print(*p);

  return 0;
}

