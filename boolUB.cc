//  c++ -O2 -Wall -fsanitize=undefined boolUB.cc

#include<memory>
#include<iostream>
int main() {

  unsigned char a[4] = {123,244,123,244};

  void * p = a;

  auto * q1 = new(p)  bool;

  std::cout << "here it comes" << std::endl;
  bool b = *q1;

  auto * q2 = new(p)  float;

  std::cout << "here it's happy" << std::endl;
  float f = *q2;


  return b&(f!=0);

}
