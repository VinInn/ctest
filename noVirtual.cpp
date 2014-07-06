#include <boost/any.hpp>
#include <boost/bind.hpp>
// #include <boost/lambda/lambda.hpp>
#include <boost/function.hpp>
// #include <boost/shared_ptr.hpp>


#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

class Me {
public:
  Me();
  
  void op();
  
  void f();
  void g();
  
  std::vector<boost::any> many;
};

//---------------------------------------

namespace countme{
  struct CountMe {
    CountMe():f(0),g(0){}
    int f;
    int g;
    void incrf() {f++;}
    void incrg() {g++;}
  };
  

  template<typename F> void apply (Me&me, F f) {
    CountMe * c = 0;
    if (!me.many.empty()) c = boost::any_cast<CountMe>(&me.many[0]);
    if (c) f(c);
  }  
  
  template<typename F> void apply (Me const &me, F f) {
    CountMe const * c = 0;
    if (!me.many.empty()) c = boost::any_cast<CountMe>(&me.many[0]);
    if (c) f(c);
  }  
  
  inline void f(Me&me) {
    apply(me,boost::bind(&CountMe::incrf,_1));
  }  
  
  inline void g(Me&me) {
    apply(me,boost::bind(&CountMe::incrg,_1));
  }  
  
  inline void dumpC(CountMe const * c) {
    std::cout << c->f << "," << c->g << std::endl;
  }  
  
  inline void dump(const Me&me) {
    apply(me,boost::bind(&dumpC,_1));
  }  
  
  void doit(Me&me) { 
    me.many.push_back(boost::any(countme::CountMe()));
  }

}


Me::Me() {
}

void Me::f() {
  countme::f(*this);

}

void Me::g() {
  countme::g(*this);


}

void Me::op() {
  f();
  g();
}



template<typename T>
std::ostream & print(std::ostream& co, T const & t, char const * sep="") {
  return co << t << sep;
}

int main() {

  Me a, b;
  countme::doit(a);

  a.f();
  a.op();

  b.g();

  countme::dump(a);
  countme::dump(b);

  return 0;

}
