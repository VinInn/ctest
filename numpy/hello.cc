#include<iostream>
extern "C" {
  void hello(const char * m) { std::cout << "hello " << m << std::endl;}
}
