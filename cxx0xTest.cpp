#include<vector>
#include<iostream>


class B {
public :
  virtual ~B()=default;
  void hello(){
    std::cout << "hello" << std::endl;
  }

};
class A : public B {
public:

  A() = default;
  A(int j): i(j){}
  A(int j, bool k) : i(j) {
    if(k) std::cout << i << std::endl;
  }
  void hello()=delete;


public:

  //  int i=0;
  int i;
};


int main() {

  A a;
  // a.hello();
  B * b = &a;
  b->hello();

  std::vector<int> v({1,2,3});


  // auto = [](double)(double x){ return x*2;};

  return 0;
}
