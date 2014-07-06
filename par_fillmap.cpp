#ifdef STDPOOL
#include <bits/c++allocator.h>
#undef ___glibcxx_base_allocator
#include <ext/pool_allocator.h>
#define ___glibcxx_base_allocator  __gnu_cxx::__pool_alloc
#endif

#include "poolTSSingleton.h" // overwrite boost default
#include <parallel/settings.h>
#include <map>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <malloc.h>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include<typeinfo>

#ifdef BOOST
#include <boost/pool/pool_alloc.hpp>
#include <boost/pool/object_pool.hpp>
//typedef boost::fast_pool_allocator<int,
typedef boost::fast_pool_allocator<std::pair<const int,int>,
      boost::default_user_allocator_new_delete,
      boost::details::pool::null_mutex
#ifdef LARGE
      ,2*1024*1024
#endif 
     > fast_alloc;
typedef boost::pool_allocator<std::pair<int,int>,
      boost::default_user_allocator_new_delete,
      boost::details::pool::null_mutex
#ifdef LARGE
      , 2*1024*1024
#endif
      > pool_alloc;

typedef boost::singleton_pool<boost::fast_pool_allocator_tag, sizeof(fast_alloc::value_type),
		       fast_alloc::user_allocator, fast_alloc::mutex, fast_alloc::next_size>
fast_alloc_singleton;

typedef boost::singleton_pool<boost::pool_allocator_tag, sizeof(fast_alloc::value_type),
			      pool_alloc::user_allocator, pool_alloc::mutex, pool_alloc::next_size>
pool_alloc_singleton;


void release() {
  fast_alloc_singleton::release_memory();
  pool_alloc_singleton::release_memory();
}

#ifdef MAP
typedef std::map<int,int, std::less<int>, fast_alloc> MI;
#else
typedef std::vector<std::pair<int,int>, pool_alloc > MI;
#endif

#else
void release() {}

#ifdef MAP
typedef std::map<int,int> MI;
#else
typedef std::vector<std::pair<int,int> > MI;
#endif
#endif


#ifdef SWAP
void fill(MI & m, int n=10000000) {
  MI l; l.swap(m);
  for (int i=0;i<n;i++)
    l.insert(l.end(),std::make_pair(i,i));
  l.swap(m);
}
#else
void fill(MI & m, int n=10000000) {
  for (int i=0;i<n;i++)
    m.insert(m.end(),std::make_pair(i,i));
}
#endif

struct M {
  M(): i(0){}
  MI m;
  size_t i;
  void fill(int n=10000000);
  bool loop();
  float meanDist();
};

void M::fill(int n) {
  ::fill(m,n);
}

#ifdef REG
bool M::loop() {
  register size_t l=i;
  register size_t s=m.size();
  while (l<s) {
#ifdef MAP
    if (m[l]!=l) return false;
#else
    if (m[l].second!=l) return false;
#endif
    l++;
  }
  i=l;
  return (i==10000000);
}
#else
bool M::loop() {
  while (i<m.size()) {
#ifdef MAP   
    if (m[i]!=i) return false;
#else
    if (m[i].second!=i) return false;
#endif
    i++;
  }
  return (i==10000000);
}
#endif


float M::meanDist() {
  double ret = 0;
  MI::const_iterator q=m.begin();
  MI::const_iterator p=q; p++;
  while(p!=m.end()) {
    ret += std::abs( long(&(*q)) - long(&(*p)));
    q = p;
    p++;
  }
  return ret/(m.size()-1);  
}

struct Y {
  double pos[3];
};

struct X {

  Y y;
};

template<typename T>
void deleteIt(T * p) { delete p;}

#ifdef BOOST
void churnX()
{
  boost::object_pool<X> p(
#ifdef LARGE
			  2*1024*104
#else
			  32
#endif
  );
  for (int i = 0; i < 10000; ++i)
  {
    X * const t = p.malloc();
    // Do something with t; don't take the time to free() it
  }
} // on function exit, p is destroyed, and all destructors for the X objects are called
#else
void churnX()
{
  std::vector<X*> p(10000,(X*)0);
  for (int i = 0; i < 10000; ++i)
  {
    p[i] =new X();
     // Do something with t; don't take the time to free() it 
  }
  for (int i = 0; i < 10000; ++i)
  {
    delete p[i];
     // Do something with t; don't take the time to free() it 
  }

} // on function exit, p is destroyed, // NO! and all destructors for the X objects are called
#endif

namespace {
  // a global one
  M & gm() {
    static M m;
    return m;
  }
}

void go(int ntimes) {

  float mean=0;
  for (int i=0; i<ntimes; i++) {
    release();
    for (int i=0; i<100; i++) {
#ifdef HEAP
      std::auto_ptr<M> p(new M);
      M & m = *p;
#else
      M m;
#endif
      
      m.fill(1000);
      mean += m.meanDist();

      churnX();
      
#ifdef LOOP
      if(!gm().loop()); // std::cout << gm.gm.size() << std::endl;
      if(!m.loop()); // std::cout << m.m.size() << std::endl;
#endif
    }


  }
  std::cout << mean/100/ntimes << std::endl;
  //  malloc_stats();
  
}


int main() {
 
  std::cout << " N th " << omp_get_max_threads() << std::endl;

  __gnu_parallel::_Settings s;
  s.for_each_minimal_n=1;
//  s.force_parallel = true;
  __gnu_parallel::_Settings::set(s);

//  std::vector<int> v(6,1000);
// __gnu_parallel::for_each(v.begin(),v.end(),boost::bind(go,_1),__gnu_parallel::parallel_omp_loop);
 //  std::for_each(v.begin(),v.end(),boost::bind(go,_1));

  gm().fill(1000);

#pragma omp parallel for
  for (int i=0; i<6; i++) go(500);

  malloc_stats();
  
  return 0;
    
}
