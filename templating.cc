#include<string>
#include<unistd.h>
#include <iostream>

class A {

public:
  template<typename T> T f(const std::string& a);

};


template<>
inline int
A::f<int>(const std::string&a) { return ::atoi(a.c_str());}
template<>
inline std::string
A::f<std::string>(const std::string&a) { return a;}


template<typename T>
class B {
public:
  B(A&ia);

  std::string a;
  int b;
} ;

template<typename T>
inline 
B<T>::B(A&ia) {
  a = ia. template f<std::string>("q");
  b = ia.f<int>("2");

}


int main() { 

  A a;
  B<double> b(a);

  std::cout << b.a << " " << b.b << std::endl;

  return 0;
}
