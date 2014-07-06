#include "Derived.h"
#include<iostream>

extern "C" void bhook();

extern "C" void bhook()
 {
  std::cout << "\nbreak point hook" << std::endl;
}

namespace {
  struct hello {
    hello() {
      bhook();
      std::cout << "\nhello D" << std::endl;
      D a(423,-423);
      a.who(a);
      a.hi();
     std::cout << std::endl;
    }
  };
#ifdef SHELLO
  hello hi;
#endif
}
