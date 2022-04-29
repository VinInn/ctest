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

  void free(int i) {
    m_used[i]=false;
  }

  int alloc(uint64_t s) {
    auto i = allocImpl(s);

    //test garbage
    if(totBytes>4507964512) garbageCollect();

    if (i>=0) {
       assert(m_used[i]);
       return i;
    } 
    garbageCollect();
    return allocImpl(s);
  }

  int allocImpl(uint64_t s) {
    auto b = poolDetails::bucket(s);
    assert(s<=poolDetails::bucketSize(b));
    int ls = size();
    // look for an existing slot
    for (int i=0; i<ls; ++i) {
      if (b!=m_bucket[i]) continue;    
      if (m_used[i]) continue;
      bool exp = false;
      if (m_used[i].compare_exchange_weak(exp,true)) {
        // verify if in the mean time the garbage collector did operate
        if(nullptr == m_slots[i]) {
          m_used[i] = false;
          continue;
        }
        return i;
      }
    }

    // try to create in existing slot (if garbage has been collected)
    ls = useOld(b);
    if (ls>=0) return ls;
    // try to allocate a new slot
    if (m_size>=maxSlots) return -1;
    ls = m_size++;
    if (ls>=maxSlots) return -1;
    return createAt(ls,b);
  }

  int createAt(int ls, int b) {
    assert(m_used[ls]);
    
    m_bucket[ls]=b;
    auto as = poolDetails::bucketSize(b);
    assert(nullptr==m_slots[ls]);
    m_slots[ls]=Traits::alloc(as);
    if (nullptr == m_slots[ls]) return -1;
    totBytes+=as;
    nAlloc++;
    return ls;
  }

  void garbageCollect() {
    int ls = size();
    for (int i=0; i<ls; ++i) {
      if (m_used[i]) continue;
      if (m_bucket[i]<0) continue; 
      bool exp = false;
      if (!m_used[i].compare_exchange_weak(exp,true)) continue;
      assert(m_used[i]);
      if( nullptr != m_slots[i]) {
        assert(m_bucket[i]>=0);  
        Traits::free(m_slots[i]);
        totBytes-= poolDetails::bucketSize(m_bucket[i]);
      }
      m_slots[i] = nullptr;
      m_bucket[i] = -1;
      m_used[i] = false; // here memory fence as well
    }
  }


  int useOld(int b) {
    int ls = size();
    for (int i=0; i<ls; ++i) {
      if ( m_bucket[i]>=0) continue;
      if (m_used[i]) continue;
      bool exp = false;
      if (!m_used[i].compare_exchange_weak(exp,true)) continue;
      if( nullptr != m_slots[i]) { // ops allocated and freed
        assert(m_bucket[i]>=0);
        m_used[i] = false;
        continue;
      }
      assert(m_used[i]);
      createAt(i,b);
      return i;
    }
    return -1;
  }

  void dumpStat() const {
   uint64_t fn=0; 
   uint64_t fs=0;
   int ls = size();
   for (int i=0; i<ls; ++i) {
      if (m_used[i]) {
        fn++;
        fs += (1<<m_bucket[i]);
      }
   }
   std::cout << "# slots " << size() << '\n'
              << "# bytes " << totBytes << '\n'
              << "# alloc " << nAlloc << '\n'
              << "# used " << fn << ' ' << fs << '\n'
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


#include<random>
#include<limits>


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

  auto start = std::chrono::high_resolution_clock::now();


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

  std::atomic<int> nt=0;

  auto test = [&] {
   int const me = nt++;
   auto delta = std::chrono::high_resolution_clock::now()-start;

   std::mt19937 eng(me+std::chrono::duration_cast<std::chrono::milliseconds>(delta).count());
   std::uniform_int_distribution<int> rgen1(1,100);
   std::uniform_int_distribution<int> rgen2(3,24);
   std::cout << "first RN " << rgen1(eng) << " at " << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count() <<  std::endl;

   int iter=0;
   while(true) {
     iter++;
     auto n = rgen1(eng);
     int ind[n];
     for (auto & i : ind) {     
       int b = rgen2(eng);
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
