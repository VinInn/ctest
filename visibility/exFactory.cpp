#include "Base.h"


extern "C" typedef Factory<Base> * factoryP();

#include <dlfcn.h>
#include <string>

factoryP * magic(std::string const & c) {
  std::string shlib("plug"+c +".so");
  std::string fname("factory"+c);
  void * dl = dlopen(shlib.c_str(),RTLD_LAZY);
  return reinterpret_cast<factoryP*>(dlsym(dl,fname.c_str()));
}

#include<iostream>
#include<typeinfo>


Factory<Base>::pointer get(std::string const & c) {

  std::cout << "Get " << c << std::endl;
  auto factory = magic(c);

  Factory<Base>::pointer bp = (*(*factory)())();

  (*bp).hi();

  std::cout << (*bp).i1() << " " << (*bp).i2()  << std::endl;
 
  std::cout << typeid(*bp).name() << " " << &typeid(*bp) << std::endl;

  return bp;

}

int main() {
  auto a =  get("A");
  auto d = get("D");
  (*a).who(*d);
  (*d).who(*a);

};
