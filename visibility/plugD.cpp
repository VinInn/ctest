#include "Derived.h"
namespace {

  struct FactoryD : public Factory<Base> {
    pointer operator()() {
      return pointer(new D(7.,-92));
    } 
    
  };

}


// extern "C" Factory<Base>* factoryA() __attribute__((vinPlug("A")));

extern "C" Factory<Base> * factoryD() {
  static FactoryD local;
  return &local;
}
