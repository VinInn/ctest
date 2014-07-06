#include <typeinfo>
#include<iostream>

#include "cxxabi.h"

void foo(const std::type_info& t) {
  char buff[16*1024];
  size_t len = sizeof(buff);
  ::memset(buff,0,len);
  int    status = 0;
  const char* rr = __cxxabiv1::__cxa_demangle(t.name(), buff, &len, &status);
  std::cout << "Called cxxabi for " << t.name() << " " << (void*)rr << " " << buff << std::endl;
}

int main() {

  double d;
  double a[3];
  double b[1][2];
  double c[1][2][3];

  std::cout << typeid(a).name() << " " << sizeof(a) << std::endl;
 std::cout << typeid(b).name() << " " << sizeof(b) << std::endl;
 std::cout << typeid(c).name() << " " << sizeof(c) << std::endl;

 foo( typeid(d)); foo( typeid(a));foo( typeid(b));foo( typeid(c));

  return 0;

}
