#include <typeinfo>
#include <iostream>
#include "cxxabi.h"

void foo(const std::type_info& t) {
  char buff[16*1024];
  size_t len = sizeof(buff);
  ::memset(buff,0,len);
  int    status = 0;
  const char* rr = __cxxabiv1::__cxa_demangle(t.name(), buff, &len, &status);
  std::cout << "Called cxxabi for " << t.name() << " " << (void*)rr << " " << buff << std::endl;
}
int main(int argc, char** argv) {
   std::cout << "This one works...." << std::endl;
   foo(typeid(void**));
   std::cout << "This one is buggy...." << std::endl;
   foo(typeid(int));
}
