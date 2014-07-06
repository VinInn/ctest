#ifndef AlgoA_H
#define AlgoA_H

#include "FW.h"
#include <memory>

class DoAAlgo {
public:
  virtual ~DoAAlgo(){}
  int operator()() { 
    do1();
    do2();
    return res;
  }

private:
  virtual void do1()=0;
  virtual void do2()=0;
protected:
  int res;
};

class DoA {
public:
  virtual ~DoA(){}
  using AlgoPtr = std::unique_ptr<DoAAlgo>;
  virtual AlgoPtr getAlgo(Ev const&, Es const&) const =0;

};

std::unique_ptr<DoA> makeDoA(int, Config const &);


#endif
