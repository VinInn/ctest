#include <vector>
// #include <boost/function.hpp>
// #include <boost/bind.hpp>

#include <iostream>


template<typename T>
T f() {
  T t;
  return t;
}

template<typename T>
struct G : public T {
  G(int k) : t(k) {}
  operator T &() { return t;}
  T t;
};


int main() {

  typedef std::vector<int> VI;
  VI v(10);

  f<VI>().swap(v);

  std::cout << v.size() << std::endl;

  G<VI> g(3);
  v.swap(g);

  std::cout << v.size() << std::endl;

  
  return 0;
}
