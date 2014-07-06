#include <boost/bind.hpp>
#include <map>
#include <string>
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <algorithm>


struct X {

  void f(int c) { 
    std::cout << c << std::endl;
  }

  int k;
};


template<typename P>
typename P::second_type & second(P & p) { return p.second;}
//template<typename T1, typename T2>
// T1 const & second(std::pair<T1,T2> const & p) { return p.second;}


int main() {


  typedef std::map<int,boost::shared_ptr<X> > M;
  M m;

  m.insert(std::make_pair(1,boost::shared_ptr<X>( new X)));
  m.insert(std::make_pair(2,boost::shared_ptr<X>( new X)));


  int k = 4;
  std::for_each(m.begin(),m.end(),boost::bind(&X::f,boost::bind(second<M::value_type>,_1),k));
  std::for_each(m.begin(),m.end(),boost::bind(&X::f,boost::bind(&M::value_type::second,_1),k));


  return 0;

}
