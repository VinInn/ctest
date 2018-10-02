#include<vector>
#include<algorithm>
#include<iostream>
namespace {
  int nm = 0;
  int nma = 0;
  int ns = 0;
  int nis = 0;
  int nd=0;
  int nc=0;
  int na=0;
}

void zero () {
  nm = 0;
  nma = 0;
  ns = 0;
  nis = 0;
  nd=0;
  nc=0;
  na=0;

}


//
// there is a bug....
//

struct A {
  
  A() : i(0), v(10000){++nd;}
  explicit A(int ii, int s=10000) : i(ii), v(s){}
  A(const A& a) : i(a.i), v(a.v) {++nc;}
  A(const A&& a) noexcept : i(a.i), v(std::move(a.v)) {++nm;}
  A& operator=(A const & a) {
    i=a.i; v=a.v; 
    ++na;
    return *this;
  }

  A& operator=(A && a) noexcept {
    i=a.i; v=std::move(a.v);
    ++nma;
    return *this;
  }


  inline void swap(A&a) {
    std::swap(i,a.i);
    v.swap(a.v);
    ++ns;
  }

  int i;
  std::vector<int> v;
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

  std::cout << "size " << a.size() << std::endl;
  std::cout << "number of swaps " << ns << " " << nis << std::endl;
  std::cout << "number of copy " << nc << " " << na << " " << nd << std::endl;
  std::cout << "number of move " << nm << " " << nma  << std::endl;

  return 0;

}
