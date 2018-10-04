#include<random>
#include<vector>
#include<cstdint>
#include<algorithm>
#include<iostream>
#include<cassert>

namespace {
  int nm = 0;
  int nma = 0;
  int ns = 0;
  int nis = 0;
  int nd=0;
  int nc=0;
  int naa=0;
}

void zero () {
  nm = 0;
  nma = 0;
  ns = 0;
  nis = 0;
  nd=0;
  nc=0;
  naa=0;

}


void print() {

  std::cout << "number of swaps " << ns << " " << nis << std::endl;
  std::cout << "number of copy " << nc << " " << naa << " " << nd << std::endl;
  std::cout << "number of move " << nm << " " << nma  << std::endl;


}

//
// there is a bug....
//

struct A {
  
  A() {++nd;}
  explicit A(int ii) : i(ii) {}
  explicit A(int ii, int s) : i(ii), v(s){}
  A(const A& a) : i(a.i), v(a.v) {++nc;}
  A(const A&& a) noexcept : i(a.i), v(std::move(a.v)) {++nm;}
  A& operator=(A const & a) {
    i=a.i; v=a.v; 
    ++naa;
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


constexpr int N=10000;
constexpr int M=40;

std::mt19937 reng;
std::poisson_distribution<int> aGen(N);
std::poisson_distribution<int> bGen(M);


void one(bool doprint) {

  int totsize=0; int totcapacity=0;


  std::vector<A> va;
  int i=0;
  // generate number of As and for each A number of contained indices
  // this is "the truth"
  auto na = aGen(reng);
  // make na even (like is easear)
  na = 2*(na/2+1);
  int nb[na];
  for(int i=0;i<na;++i) nb[i]=bGen(reng);
  try {
    // here we fake a clustering process
    // first we assume some how in each A associate twice as much elements
    auto nah = na/2;
    for(int i=0;i<nah;++i) {
      va.push_back(A(i)); 
      auto & v = va.back().v;
      for(int j=0;j<nb[i];++j) v.push_back(j);
      assert(i+nah<na);
      for(int j=0;j<nb[i+nah];++j) v.push_back(j);
      assert(v.size()==nb[i]+nb[i+nah]);
    }
    assert(va.back().i==nah-1);
    // now we split
    for(int i=0;i<nah;++i) {
      va.push_back(A(i+nah));
      auto & v1 = va[i].v;
      auto & v2 = va.back().v;
      // insert in v2 what belongs there
      for(int j=nb[i];j<v1.size();++j) v2.push_back(v1[j]);
      // remove them from v1
      v1.resize(nb[i]);
    }
    assert(va.size()==na);
    for(int i=0;i<na;++i) {
      assert(va[i].i==i);
      assert(va[i].v.size()==nb[i]);
      totsize+=va[i].v.size();
      totcapacity+=va[i].v.capacity();
    }

    /// bonus sort!
    std::sort(va.begin(),va.end(),
	      [](auto const & a, auto const & b) { return a.v.size()<b.v.size();}
	      );
    
  }
  catch(...) {
    std::cout << "oops" << std::endl;    
  }

  if (doprint) {
    std::cout << "a size / capacity " << va.size() << ' ' << va.capacity() << std::endl;
    std::cout << "tot size / capacity " << totsize << ' ' << totcapacity << std::endl;
    print();
  }
}


int main() {

  one(true);

  for (int i=0; i<100; ++i)
    one(false);

  one(true);

  return 0;

}
