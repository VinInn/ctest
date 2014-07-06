#ifdef V_H
#include<vector>
#endif

class S;

class A : public std::vector<int> {
public:
  explicit A(S const& s); 
  ~A();
  
private:
  S * s;
  
};


#include <typeinfo>
template<typename T>
struct P {
  static void print() {
    std::cout << typeid(T).name() < std::endl;
  }

};


class B {
 public:
  B();

  void print();
 private:
  
  A a;
  
};

// this is S.h
#ifdef S_H

  class S {
  public:
  S(const char *);
  
  };
#endif 
  
// #include <typeinfo>
// #include <iostream>


B::B(): a("in B"){
  std::vector<int> v;
  a.push_back(1);
  a.swap(v);
}


void B::print() {
  // P<S>::print();
}
