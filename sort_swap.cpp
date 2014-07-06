#include<vector>
#include<algorithm>
#include<iostream>

/*
namespace std {
  template<typename _ForwardIterator>
  inline void
  iter_swap<_ForwardIterator, _ForwardIterator>(_ForwardIterator _a, _ForwardIterator _b) {
    std::swap(*_a,*_b);
  }
}
*/

/*
namespace std {
  template<typename T>
  inline void
  iter_swap<typename std::vector<T>::iterator, typename std::vector<T>::iterator>(typename std::vector<T>::iterator __a, typename std::vector<T>::iterator __b) {
    using std::swap;
    swap(*__a,*__b);
    // ++nis;
  }
}
*/

namespace {
  int ns = 0;
  int nis = 0;
  int nd=0;
  int nc=0;
  int na=0;
  int nm=0;
  int am=0;
}

void zero () {
  ns = 0;
  nis = 0;
  nd=0;
  nc=0;
  na=0;
  nm=0;
  am=0;

}

struct A {
  
  A() : i(0), v(100){++nd;}
  explicit A(int ii, int s) : i(ii), v(s){}
  A(const A& a) : i(a.i), v(a.v) {++nc;}

  A& operator=(A const & a) {
    i=a.i; v=a.v; 
    ++na;
    return *this;
  }

#if defined( __GXX_EXPERIMENTAL_CXX0X__)

  A(A&& a) : i(a.i), v(std::move(a.v)) {++nm;}

  A& operator=(A && a) {
    i=a.i; v.swap(a.v); 
    ++am;
    return *this;
  }
#endif

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

#ifdef SWAP 
// namespace std {
//  template<> 
inline void swap(A & rh, A & lh) {
    rh.swap(lh);
}

// }
#endif


#ifdef ITSWAP 
namespace std {
  template<>
  inline void
  iter_swap<std::vector<A>::iterator, std::vector<A>::iterator>(std::vector<A>::iterator __a, std::vector<A>::iterator __b) {
    using std::swap;
    swap(*__a,*__b);
    ++nis;
  }
}
#endif

struct Set {
  Set(int ii=0) : i(ii){}
  int i;
  void operator()(A& a) {
    a.i=i++;
  }
};

#include "RealTime.h"

void go(int is) {

  {
    zero();
    perftools::TimeType start = perftools::realTime();
    std::vector<A> a; 
    // a.reserve(10000);
    for(int i=0; i<10000;i++)
      a.push_back(A(i,100));
    perftools::TimeType end = perftools::realTime();
    std::cout << "push_back tot real time " << 1.e-9*(end-start) << std::endl;
    
    std::cout << "number of swaps " << ns << " " << nis << std::endl;
    std::cout << "number of copy " << nc << " " << na << " " << nd << std::endl;
    std::cout << "number of move " << nm << " " << am << std::endl;
    
  }


  std::vector<A> a(10000);
  std::for_each(a.begin(),a.end(),Set(is));
  {
    zero();
    perftools::TimeType start = perftools::realTime();
    std::random_shuffle(a.begin(),a.end());
    std::sort(a.begin(),a.end());
    perftools::TimeType end = perftools::realTime();
    std::cout << "shuffle and sort tot real time " << 1.e-9*(end-start) << std::endl;
    
   std::cout << "number of swaps " << ns << " " << nis << std::endl;
   std::cout << "number of copy " << nc << " " << na << " " << nd << std::endl;
   std::cout << "number of move " << nm << " " << am << std::endl;
  }

  {
    zero();
    perftools::TimeType start = perftools::realTime();
    std::random_shuffle(a.begin(),a.end());
    std::stable_sort(a.begin(),a.end());
    perftools::TimeType end = perftools::realTime();
    std::cout << "shuffle and stable sort tot real time " << 1.e-9*(end-start) << std::endl;
    std::cout << "number of swaps " << ns << " " << nis << std::endl;
    std::cout << "number of copy " << nc << " " << na << " " << nd << std::endl;    
    std::cout << "number of move " << nm << " " << am << std::endl;

  }
}

int main() {

  go(0);
  go(32000);
  go(123456789);

  return 0;
};
