#include <cmath>
#include <iostream>
#include <boost/thread/thread.hpp>
#include <boost/ref.hpp>
#include <memory>
#include <vector>
#include <algorithm>
#include <boost/bind.hpp>

template<typename F>
class FutureBase {
public:
  virtual ~FutureBase(){}
  virtual F const & result() const=0;
 
};


template<typename F>
class FutureAsync : public FutureBase<F> {
public:

  FutureAsync(F & fi) : f(fi),
			t(boost::ref(f)) {}

  F const & result() const {
    t.join();
    return f;
  } 
  
private:
  F & f;
  mutable boost::thread t;
};

template<typename F>
class FutureSync : public FutureBase<F>  {
public:

  FutureSync(F & fi) : f(fi),
		       done(false) {}

  F const & result() const {
    if(!done) { 
      f();
      done=true;
    }
    return f;
  } 
  
private:
  mutable F & f;
  mutable bool done;
};


inline bool & futureSwitch() {
  static bool m_async=true;
  return m_async;
}



template<typename F>
class Future : public FutureBase<F>  {
public:
  typedef FutureBase<F> base;

  Future(F & fi) : f(futureSwitch() ? (base*)(new FutureAsync<F>(fi)) : 
		     (base*)(new FutureSync<F>(fi))) {}

  F const & result() const {
    return (*f).result();
  } 
  
private:
  std::auto_ptr< FutureBase<F> > f;
};


struct Nop {

  void operator()(){}

};

struct Hello {
  bool done;
  Hello() : done(false) {};
  void operator()(){done=true;}

};

struct Sort {

  explicit Sort(size_t s=10000) : v(s) {
    std::generate(v.begin(),v.end(),boost::bind(std::rand));
  }

  void operator()(){
    std::sort(v.begin(),v.end());
  }

  std::vector<int> v; 
};


#include "RealTime.h"


int main(int argc) {
  
  futureSwitch() = argc>1;
  std::cout << ( futureSwitch() ? "thread" : "sequential" ) << std::endl;

  {    
    Nop nop;
    perftools::TimeType start = perftools::realTime();
    Future<Nop> f(nop);
    
    const Nop & res = f.result();
    
    perftools::TimeType end = perftools::realTime();
    std::cout << "NOP   tot real time " << 1.e-9*(end-start) << std::endl;
  }

  {    
    Nop nop;
    perftools::TimeType start = perftools::realTime();
    Future<Nop> f(nop);
    
    const Nop & res = f.result();
    
    perftools::TimeType end = perftools::realTime();
    
    std::cout << "NOP   tot real time " << 1.e-9*(end-start) << std::endl;
  }
 {    
    Nop nop1;
    Nop nop2;
    perftools::TimeType start = perftools::realTime();
    Future<Nop> f1(nop1);
    Future<Nop> f2(nop2);
    
    const Nop & res1 = f1.result();
    const Nop & res2 = f2.result();
    
    perftools::TimeType end = perftools::realTime();
    
    std::cout << "2 NOP tot real time " << 1.e-9*(end-start) << std::endl;
  }

  {    
    Hello hi;
    perftools::TimeType start = perftools::realTime();
    Future<Hello> f(hi);
    
    const Hello & res = f.result();
    
    perftools::TimeType end = perftools::realTime();
    
    std::cout << "Hello tot real time " << 1.e-9*(end-start) << std::endl;
    if (!res.done) std::cout << "error " << std::endl;
    if (!hi.done) std::cout << "error " << std::endl;
  }

 {
    
    Sort s;

    perftools::TimeType start = perftools::realTime();

    Future<Sort> f(s);
    
    const Sort & res = f.result();
    
    perftools::TimeType end = perftools::realTime();
    
    std::cout << "Sort  tot real time " << 1.e-9*(end-start) << std::endl;
    if (res.v.front()>res.v.back()) std::cout << "error " << std::endl;
  }
 {
    
    Sort s;

    perftools::TimeType start = perftools::realTime();

    Future<Sort> f(s);
    
    const Sort & res = f.result();
    
    perftools::TimeType end = perftools::realTime();
    
    std::cout << "Sort  tot real time " << 1.e-9*(end-start) << std::endl;
    if (res.v.front()>res.v.back()) std::cout << "error " << std::endl;
  }
 {
    
   Sort s(100000);

    perftools::TimeType start = perftools::realTime();

    Future<Sort> f(s);
    
    const Sort & res = f.result();
    
    perftools::TimeType end = perftools::realTime();
    
    std::cout << "Sort 100 tot real time " << 1.e-9*(end-start) << std::endl;
    if (res.v.front()>res.v.back()) std::cout << "error " << std::endl;
  }
 {
    
    Sort s1;
    Sort s2;

    perftools::TimeType start = perftools::realTime();

    Future<Sort> f1(s1);
    Future<Sort> f2(s2);
    
    const Sort & res1 = f1.result();
    
    const Sort & res2 = f2.result();
    
    perftools::TimeType end = perftools::realTime();
    
    std::cout << "2 Sorts tot real time " << 1.e-9*(end-start) << std::endl;
    if (res1.v.front()>res1.v.back()) std::cout << "error " << std::endl;
    if (res2.v.front()>res2.v.back()) std::cout << "error " << std::endl;
  }
 {
    
    Sort s1(100000);
    Sort s2(100000);

    perftools::TimeType start = perftools::realTime();

    Future<Sort> f1(s1);
    Future<Sort> f2(s2);
    
    const Sort & res1 = f1.result();
    
    const Sort & res2 = f2.result();
    
    perftools::TimeType end = perftools::realTime();
    
    std::cout << "2 Sorts 100 tot real time " << 1.e-9*(end-start) << std::endl;
    if (res1.v.front()>res1.v.back()) std::cout << "error " << std::endl;
    if (res2.v.front()>res2.v.back()) std::cout << "error " << std::endl;
  }

}

