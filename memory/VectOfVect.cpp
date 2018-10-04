#include<random>
#include<vector>
#include<cstdint>
#include<algorithm>
#include<iostream>
#include<cassert>

#include<chrono>

#include "memory_usage.h"

auto start = std::chrono::high_resolution_clock::now();

uint64_t maxLive=0;

void stop(const char * m) {
  auto delta = std::chrono::high_resolution_clock::now()-start;
  maxLive= std::max( maxLive, memory_usage::totlive() );
  std::cout << m;
  std::cout << " elapsted time " << std::chrono::duration_cast<std::chrono::nanoseconds>(delta).count() << std::endl;
  std::cout << "allocated so far " << memory_usage::allocated();
  std::cout << " deallocated so far " << memory_usage::deallocated() << std::endl;
  std::cout << "total / max live " << memory_usage::totlive() << ' ' << maxLive << std::endl;

  start = std::chrono::high_resolution_clock::now();
}




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
  int totElement=0;
  for(int i=0;i<na;++i) totElement += (nb[i]=bGen(reng));
  try {
    stop("before first loop");
    // here we fake a clustering process
    // WHAT WE KNOW IS ONLY THAT WE HAVE totElement ELEMENTS
    // they are store somewhere we do not deal with the real algo here, only data structure
    // representing their grouping
    // we assume that the final results is "na" groups each associated with nb[i] elements.
    // the solution can be any data structure not necessarely the "vector of vector of indices used here
    // the only fixed assumpition is that in the first loop somehow each A is associated to twice as much elements
    // the asserts shall be respected
    auto nah = na/2;
    auto kk1=0;  // fakes the index (pointer) to an element
    auto kk2=0; for(int i=0;i<nah;++i) kk2+=nb[i]; // same for the other half
    for(int i=0;i<nah;++i) {
      va.push_back(A(i)); 
      auto & v = va.back().v;
      for(int j=0;j<nb[i];++j) v.push_back(kk1++);
      for(int j=0;j<nb[i+nah];++j) v.push_back(kk2++);

    }
    stop("after first loop");
 
    assert(va.size()==nah);
    for(int i=0;i<nah;++i) {
      assert(va[i].v.size()==nb[i]+nb[i+nah]);
    }

    stop("before second loop");
    // now we split
    // again algo irrelevant, just data structure and their filling to be optimized
    for(int i=0;i<nah;++i) {
      va.push_back(A(i+nah));
      auto & v1 = va[i].v;
      auto & v2 = va.back().v;
      // insert in v2 what belongs there
      for(int j=nb[i];j<v1.size();++j) v2.push_back(v1[j]);
      // remove them from v1
      v1.resize(nb[i]);
    }
    stop("after second loop");

    int kk=0;
    assert(va.size()==na);
    for(int i=0;i<na;++i) {
      assert(va[i].i==i);
      assert(va[i].v.size()==nb[i]);
      for (auto e : va[i].v) { assert(e==kk); ++kk;}
      totsize+=va[i].v.size();
      totcapacity+=va[i].v.capacity();
    }

    /// bonus sort!
    stop("before sort");
    std::sort(va.begin(),va.end(),
	      [](auto const & a, auto const & b) { return a.v.size()<b.v.size();}
	      );
    stop("after sort");
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

  for (int i=0; i<100; ++i) {
    one(false);
    stop("after call");

  }

  one(true);

  stop("\nat the end");

  
  return 0;

}
