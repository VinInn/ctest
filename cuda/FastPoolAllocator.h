#pragma once

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
    // if(totBytes>4507964512) garbageCollect();

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
        nFree++;
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
              << "# free " << nFree << '\n'
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
  std::atomic<uint64_t> nFree = 0;

};




#include <cstdlib>
struct PosixAlloc {

  using Pointer = void *;

  static Pointer alloc(size_t size) { return ::malloc(size); }
  static void free(Pointer ptr) { ::free(ptr); }

};

#ifdef __CUDACC__
#include<cuda.h>
#include<cuda_runtime.h>
#include <cuda_runtime_api.h>

struct CudaDeviceAlloc {

  using Pointer = void *;

  static Pointer alloc(size_t size) { Pointer p; auto err = cudaMalloc(&p,size); return err==cudaSuccess ? p : nullptr;}
  static void free(Pointer ptr) { cudaFree(ptr); }

};

struct CudaHostAlloc {

  using Pointer = void *;

  static Pointer alloc(size_t size) { Pointer p; auto err = cudaMallocHost(&p,size); return err==cudaSuccess ? p : nullptr;}
  static void free(Pointer ptr) { cudaFreeHost(ptr); }

};
#endif

