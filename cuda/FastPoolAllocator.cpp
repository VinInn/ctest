#include<atomic>
#include<array>
#include<memory>
#include<algorithm>
#include<cassert>
#include<mutex>
#include <vector>
#include <cstdint>
#include<iostream>


namespace poolDetails {

 constexpr int bucket(uint64_t s) { return 64-__builtin_clzl(s-1); }
 constexpr uint64_t bucketSize(int b) { return 1LL<<b;}

};

template<typename Traits, int N>
class FastPoolAllocator {

public:


  
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
    std::cout << "not found " << std::endl;
    if (m_size>=maxSlots) return -1;
    ls = m_size++;
    std::cout << "create " << ls << std::endl;
    if (ls>=maxSlots) return -1;
    m_used[ls]=true;
    m_bucket[ls]=b;
    std::cout << "alloc "  << s << ' ' << b << ' ' << poolDetails::bucketSize(b) << std::endl;
    m_slots[ls]=Traits::alloc(poolDetails::bucketSize(b));
    std::cout << "at " << m_slots[ls] << std::endl;
    if (nullptr == m_slots[ls] ) return -1;
    return ls;
  }

  void free(int i) {
    m_used[i]=false;
  }

private:

  std::vector<int> m_bucket = std::vector<int>(maxSlots);
  std::vector<std::atomic<Pointer>> m_slots = std::vector<std::atomic<Pointer>>(maxSlots);
  std::vector<std::atomic<bool>> m_used = std::vector<std::atomic<bool>>(maxSlots);
  std::atomic<int> m_size=0;
};




#include <cstdlib>
struct PosixAlloc {

  using Pointer = void *;

  static Pointer alloc(size_t size) { return ::malloc(size); }
  static void free(Pointer ptr) { ::free(ptr); }

};





int main() {


  FastPoolAllocator<PosixAlloc,1024> pool;
  assert(0==pool.size());

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

  return 0;

}
