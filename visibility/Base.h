#ifndef BASE_H
#define BASE_H

#include "macros.h"

class Base {
public:
  Base(int ii, float ff);
  /* inlined will not work as n_s is hidden
  : i(ii), f(ff) {
  ++n_s;
  }
  */
  virtual ~Base();

  virtual int i1() const=0;

  int i2() const {
    return ii();
  }

  virtual void hi() const=0;

  virtual void who(Base const & b) const=0;

private:

  virtual int ii() const dso_internal  { return i;}

  int i;
  float f;
public:
  static int dso_internal n_s;

};

#include<memory>
#include<algorithm>
template<typename B>
class Factory {
public:
  typedef std::shared_ptr<B> pointer;

  virtual pointer operator()()=0;

  /*
  template<typename T, typename... Args>
  pointer operator()(Args&&... args) {
    return pointer(new T(std::forward<Args>(args)...));
  }
  */

};


#endif
