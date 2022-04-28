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
    if (m_size>=maxSlots) return -1;
    ls = m_size++;
    if (ls>=maxSlots) return -1;
    m_used[ls]=true;
    m_bucket[ls]=b;
    m_slots[ls]=Traits::alloc(poolDetails::bucketSize(b));
    if (nullptr == m_slots[ls] ) return -1;
    return ls;
  }

  void free(int i) {
    m_used[i]=false;
  }

private:

  std::array<int,maxSlots> m_bucket;
  std::array<std::atomic<Pointer>,maxSlots> m_slots;
  std::array<std::atomic<bool>,maxSlots> m_used;
  std::atomic<int> m_size=0;
};




#include <cstdlib>
struct PosixAlloc {

  using Pointer = void *;

  static Pointer alloc(size_t size) { return ::malloc(size); }
  static void free(Pointer ptr) { ::free(ptr); }

};





int main() {


  FastPoolAllocator<PosixAlloc,1024*1024> pool;
  assert(0==pool.size());

  int s = 40;

  int i0 = pool.alloc(s);
  assert(1==pool.size());
  assert(i0>=0);
  auto p0 = pool.pointer(i0);
  assert(nullptr!=p0);

  return 0;

}
