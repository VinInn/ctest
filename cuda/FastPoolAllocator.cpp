#include<atomic>
#include<array>
#include<memory>
#include<algorithm>
#include<cassert>
#include<mutex>
#include <vector>
#include <cstdint>
#include<iostream>
#include<chrono>

namespace poolDetails {

 constexpr int bucket(uint64_t s) { return 64-__builtin_clzl(s-1); }
 constexpr uint64_t bucketSize(int b) { return 1LL<<b;}

};

template<typename Traits, int N>
class FastPoolAllocator {

public:

  FastPoolAllocator() {
    for ( auto & p : m_slots) p = nullptr;
    for ( auto & p : m_used) p = true;
  }
  
  static constexpr int maxSlots = N;
  using Pointer = typename Traits::Pointer;

  int size() const { return m_size;}

  Pointer pointer(int i) const { return m_slots[i]; }

  int alloc(uint64_t s) {
    auto b = poolDetails::bucket(s);
    assert(s<=poolDetails::bucketSize(b));
    int ls = size();
    for (int i=0; i<ls; ++i) {
      if (b!=m_bucket[i]) continue;    
      if (m_used[i]) continue;
      bool exp = false;
      if (m_used[i].compare_exchange_weak(exp,true)) return i;
    }
    if (m_size>=maxSlots) return -1;
    ls = m_size++;
    if (ls>=maxSlots) return -1;
    assert(true==m_used[ls]);
    m_bucket[ls]=b;
    auto as = poolDetails::bucketSize(b);
    assert(nullptr==m_slots[ls]);
    m_slots[ls]=Traits::alloc(as);
    if (nullptr == m_slots[ls]) return -1;
    totBytes+=as;
    nAlloc++;
    return ls;
  }

  void free(int i) {
    m_used[i]=false;
  }


  void dumpStat() const {

    std::cout << "# slots " << size() << '\n'
              << "# bytes " << totBytes << '\n'
              << "# alloc " << nAlloc << '\n'
              << std::endl;
  }
 

private:

  std::vector<int> m_bucket = std::vector<int>(maxSlots,-1);
  std::vector<std::atomic<Pointer>> m_slots = std::vector<std::atomic<Pointer>>(maxSlots);
  std::vector<std::atomic<bool>> m_used = std::vector<std::atomic<bool>>(maxSlots);
  std::atomic<int> m_size=0;

  std::atomic<uint64_t> totBytes = 0;
  std::atomic<uint64_t> nAlloc = 0;
  

};




#include <cstdlib>
struct PosixAlloc {

  using Pointer = void *;

  static Pointer alloc(size_t size) { return ::malloc(size); }
  static void free(Pointer ptr) { ::free(ptr); }

};


#include<cmath>
#include<unistd.h>

#include<thread>

typedef std::thread Thread;
typedef std::vector<std::thread> ThreadGroup;
typedef std::mutex Mutex;
typedef std::lock_guard<std::mutex> Lock;


struct Node {
  int it=-1;
  int i=-1;
  void * p=nullptr;
  std::atomic<int> c=0;

};

int main() {


  FastPoolAllocator<PosixAlloc,1024*1024> pool;
  assert(0==pool.size());


  Thread monitor([&]{while(true){sleep(5); pool.dumpStat();}});
  monitor.detach();  

  int s = 40;

  std::cout << "try to allocate " << s << std::endl;

  int i0 = pool.alloc(s);
  assert(1==pool.size());
  assert(i0>=0);
  auto p0 = pool.pointer(i0);
  assert(nullptr!=p0);

  pool.free(i0);
  assert(1==pool.size());

  int i1 = pool.alloc(s);
  assert(1==pool.size());
  assert(i1==i0);
  auto p1 = pool.pointer(i1);
  assert(p1==p0);

  auto start = std::chrono::high_resolution_clock::now();

  std::atomic<int> nt=0;

  auto test = [&] {
   int const me = nt++;
   int iter=0;
   while(true) {
     iter++;
     auto delta = std::chrono::high_resolution_clock::now()-start;
     int n =  1+ int64_t(std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()/50.)%100;
     int ind[n];
     for (auto & i : ind) {     
       auto delta = std::chrono::high_resolution_clock::now()-start;
       int b = 3 + int64_t(std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()/50.)%20;
       uint64_t s = 1<<b;
       assert(s>0);
       i = pool.alloc(s+sizeof(Node));
       if (i<0) {
         std::cout << "failed at " << iter << std::endl;
         pool.dumpStat();
         return;
       }
       assert(i>=0);
       auto p = pool.pointer(i);
       assert(nullptr!=p);
       // do something???
       auto n = (Node*)(p);
       n->it = me;
       n->i = i;
       n->p = p;
       n->c=1;
     }
     for (auto i : ind) {
      auto p = pool.pointer(i);
      assert(nullptr!=p);
      auto n = (Node*)(p);
      n->c--;
      assert(n->it == me);
      assert(n->i == i);
      assert(n->p == p);
      assert(0 == n->c);
      pool.free(i);
     }
   }
  };


  const int NUMTHREADS=24;
  ThreadGroup threads;
  threads.reserve(NUMTHREADS);

   for (int i=0; i<NUMTHREADS; ++i) {
      threads.emplace_back(test);
    }

    for (auto & t : threads) t.join();

    threads.clear();
  pool.dumpStat();

  return 0;

}
