#include "AlgoB.h"



class DoB1 : public DoB {
public:
  DoB1(Config const & c) : p1(c[0]),p2(c[1]){}
  ~DoB1(){}
  AlgoPtr getAlgo(Ev const&, Es const&, Region const &) const;

private:
  friend class DoB1Algo;
  int p1;
  int p2;

};

class DoB1Algo final : public DoBAlgo {
public:
  virtual ~DoB1Algo(){}
 
private:
  friend class DoB1;
  DoB1Algo(DoB1 const & pi, Ev const& ev, Es const& es, Region const & ir) : 
    params(pi), reg(ir) {
    e1 = ev.get(params.p1);
    s1 = es.get(params.p2);
  }

  virtual int do1(){  return res=e1;}
  virtual int do2(){ return res*s1;}
 
  DoB1 const & params;
  Region const & reg;
  int e1;
  int s1;

  int res;

};

DoB1::AlgoPtr DoB1::getAlgo(Ev const&ev, Es const&es, Region const & it) const {
  return AlgoPtr(new DoB1Algo(*this,ev,es, it));
}

////
#include "AlgoA.h"

class DoB2 : public DoB {
public:
  DoB2(Config const & c) :  doA(std::move(makeDoA(c[0],c))),
  p0(c[1]), p1(c[2]), p2(c[3]), p3(c[4]){}
  ~DoB2(){}
  AlgoPtr getAlgo(Ev const&, Es const&, Region const &) const;

private:
  friend class DoB2Algo;
  std::unique_ptr<DoA> doA;
  int p0;
  int p1;
  int p2;
  int p3;

};

class DoB2Algo final : public DoBAlgo {
public:
  virtual ~DoB2Algo(){}
 
private:
  friend class DoB2;
  DoB2Algo(DoB2 const & pi, Ev const& ev, Es const& es, Region const & ir) : 
    params(pi), reg(ir), doA(std::move(params.doA->getAlgo(ev,es))) {
    s1 = ev.get(params.p0);
    e1 = es.get(params.p1);
    e2 = es.get(params.p2);
  }

  virtual int do1(){ return e1*s1;}
  virtual int do2(){ return (e2>params.p3) ? e2+(*doA)() : 0;}
 
  DoB2 const & params;
  Region const & reg;
  DoA::AlgoPtr doA;

  int e1;
  int e2;
  int s1;

};

DoB2::AlgoPtr DoB2::getAlgo(Ev const&ev, Es const&es, Region const & ir) const {
  return AlgoPtr(new DoB2Algo(*this,ev,es,ir));
}


std::unique_ptr<DoB> makeDoB(int w, Config const & c) {
  return (w==1) ? 
    std::unique_ptr<DoB>( new DoB1(c) ) :
    std::unique_ptr<DoB>( new DoB2(c) ) ;
}
