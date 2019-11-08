#include<vector>
#include<algorithm>
#include<iostream>

struct Count {
  long long nm = 0;
  long long nma = 0;
  long long ns = 0;
  long long nis = 0;
  long long nd=0;
  long long nc=0;
  long long na=0;

  void zero () {
  nm = 0;
  nma = 0;
  ns = 0;
  nis = 0;
  nd=0;
  nc=0;
  na=0;
  }


  void print() const {
    std::cout << "number of def constr " << nd << std::endl;
    std::cout << "number of swaps " << ns << " " << nis << std::endl;
    std::cout << "number of copy " << nc << " " << na << std::endl;
    std::cout << "number of move " << nm << " " << nma  << std::endl;
  }

};


   Count ca;
   Count cb;

//
// there is a bug....
//

struct B {

  B() : i(0) {++cb.nd;}
  B(int ii) : i(ii) {}
  B(const B& a) : i(a.i) {++cb.nc;}
  B(const B&& a) noexcept : i(a.i) {++cb.nm;}
  B& operator=(B const & a) {
    i=a.i;
    ++cb.na;
    return *this;
  }

  B& operator=(B && a) noexcept {
    i=a.i;
    ++cb.nma;
    return *this;
  }


  inline void swap(B&a) {
    std::swap(i,a.i);
    ++cb.ns;
  }

  int i;
};


//
// there is a SERIOUS bug....
//
struct A {
  
  A() : i(0), v(10000){++ca.nd;}
  explicit A(int ii, int s=10000) : i(ii), v(s){}
  A(const A& a) : i(a.i), v(a.v) {++ca.nc;}
  A(const A&& a) noexcept : i(a.i), v(std::move(a.v)) {++ca.nm;}
  A& operator=(A const & a) {
    i=a.i; v=a.v; 
    ++ca.na;
    return *this;
  }

  A& operator=(A && a) noexcept {
    i=a.i; v=std::move(a.v);
    ++ca.nma;
    return *this;
  }


  inline void swap(A&a) {
    std::swap(i,a.i);
    v.swap(a.v);
    ++ca.ns;
  }

  int i;
  std::vector<B> v;
};

inline bool operator<(A const& rh, A const& lh) {
  return rh.i<lh.i;
}

int main() {

  int N=400000;

  std::vector<A> a;
  // a.reserve(N);
  int i=0;
  try {
    for(;i<N;++i) 
      a.push_back(A(i));
  }
  catch(...) {
    std::cout << "oops" << std::endl;    
  }

  std::cout << "stat for A" << std::endl;
  std::cout << "size " << a.size() << std::endl;
  ca.print();
  std::cout << "stat for B" << std::endl;
  cb.print();

  return 0;

}
