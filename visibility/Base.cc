#include "Base.h"

int Base::n_s=0;

Base::Base(int ii, float ff) : i(ii), f(ff) {
    ++n_s;
}

Base::~Base() {
  --n_s;
}



#include<iostream>

namespace {
  struct dump {
    dump() {
      std::cout << "Hi from dump Base " << Base::n_s << std::endl;
    }
    ~dump() {
      std::cout << "finally left " << Base::n_s << " object" << std::endl;
    }
  };
  dump d;
}
