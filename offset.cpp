#define offsetof(type, member)  __builtin_offsetof (type, member)

class A {

public:
  int i;
  double k;
};

class B {
public:

  // virtual ~B(){}
  A a;
  int w; 

};


static constexpr int kaB_o() { return offsetof(B,a) + offsetof(A,k);}

#include<iostream>
int main() {

  constexpr auto a1 = kaB_o();

  std::cout << offsetof(A,i) << std::endl;
  std::cout << offsetof(A,k) << std::endl;
  B b;
  std::cout << offsetof(B,w) << std::endl;
  std::cout << offsetof(B,a) << std::endl;

  std::cout << a1 << std::endl;	

}
