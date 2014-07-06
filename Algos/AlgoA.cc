#include "AlgoA.h"
#include <memory>


class DoA1 : public DoA {
public:
  DoA1(Config const & c) : p1(c[0]),p2(c[1]){}
  ~DoA1(){}
  AlgoPtr getAlgo(Ev const&, Es const&) const;

private:
  friend class DoA1Algo;
  int p1;
  int p2;

};

class DoA1Algo final : public DoAAlgo {
public:
  virtual ~DoA1Algo(){}
 
private:
  friend class DoA1;
  DoA1Algo(DoA1 const & pi, Ev const& ev, Es const& es) : 
    params(pi) {
    e1 = ev.get(params.p1);
    s1 = es.get(params.p2);
  }

  virtual void do1(){ res=e1;}
  virtual void do2(){ res*=s1;}
 
  DoA1 const & params;

  int e1;
  int s1;

};

DoA1::AlgoPtr DoA1::getAlgo(Ev const&ev, Es const&es) const {
  return AlgoPtr(new DoA1Algo(*this,ev,es));
}

////

class DoA2 : public DoA {
public:
  DoA2(Config const & c) : 
  p0(c[0]), p1(c[1]), p2(c[2]), p3(c[3]){}
  ~DoA2(){}
  AlgoPtr getAlgo(Ev const&, Es const&) const;

private:
  friend class DoA2Algo;
  int p0;
  int p1;
  int p2;
  int p3;

};

class DoA2Algo final : public DoAAlgo {
public:
  virtual ~DoA2Algo(){}
 
private:
  friend class DoA2;
  DoA2Algo(DoA2 const & pi, Ev const& ev, Es const& es) : 
    params(pi) {
    s1 = ev.get(params.p0);
    e1 = es.get(params.p1);
    e2 = es.get(params.p2);
  }

  virtual void do1(){ res=e1*s1;}
  virtual void do2(){ if (e2>params.p3) res+=e2;}
 
  DoA2 const & params;

  int e1;
  int e2;
  int s1;

};

DoA2::AlgoPtr DoA2::getAlgo(Ev const&ev, Es const&es) const {
  return AlgoPtr(new DoA2Algo(*this,ev,es));
}


std::unique_ptr<DoA> makeDoA(int w, Config const & c) {
  return (w==1) ? 
    std::unique_ptr<DoA>( new DoA1(c) ) :
    std::unique_ptr<DoA>( new DoA2(c) ) ;
}
