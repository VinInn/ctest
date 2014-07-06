#ifndef Derived_H
#define Derived_H
#include "Base.h"

#pragma visibility push(internal)
class D : public Base {
public:
  D(float a, float b) :
  Base (a, b) {}

  virtual ~D();

  int i1() const { return 77;}
  
  void hi() const;

  void who(Base const & b) const;
private:
  
  virtual int ii() const dso_internal  { return -77;}
  
};
#pragma visibity pop


#endif
