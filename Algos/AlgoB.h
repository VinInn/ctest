#ifndef AlgoB_H
#define AlgoB_H
#include "FW.h"
#include <memory>

struct Region {

};

class DoBAlgo {
public:
  virtual ~DoBAlgo(){}

  virtual int do1()=0;
  virtual int do2()=0;


};

class DoB {
public:
  virtual ~DoB(){}
  using AlgoPtr = std::unique_ptr<DoBAlgo>;
  virtual AlgoPtr getAlgo(Ev const&, Es const&, Region const &) const =0;

};

std::unique_ptr<DoB> makeDoB(int, Config const &);


#endif
