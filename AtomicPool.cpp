#include<atomic>
#include<cstdint>
#include<algorithm>
#include<cassert>
#include<vector>
#include<thread>

template<typename T, int N>
struct AtomicPool {
   using Self = AtomicPool<T,N>;
   inline static Self me;
   static Self & pool() {
     return me;
   }  
   
   AtomicPool() {
    for ( auto & p : cont) p=nullptr;
   }
   ~AtomicPool() {
    for ( auto & p : cont) delete p;
   }

   struct Sentry {
     Sentry() {
       auto & pool = Self::pool();
       int n = std::min(N,pool.n.load(std::memory_order_relaxed));
       p = nullptr;
       for (int i=0; i<n; ++i) {
          p = pool.cont[i].exchange(nullptr,std::memory_order_acq_rel);  // previous changes in (*p) should become visible here
         if (p) return;
       }
       p = new T();
     }
     ~Sentry() {
       auto & pool = Self::pool();
       int n = std::min(N,pool.n.load(std::memory_order_relaxed));
       for (int i=0; i<n; ++i) {
         T * exp = nullptr;
         if (pool.cont[i].compare_exchange_weak(exp,p,std::memory_order_relaxed)) return;
       }
       n = pool.n++;
       if (n<N) {
         T * exp = nullptr;
         if (pool.cont[n].compare_exchange_weak(exp,p,std::memory_order_relaxed)) return;
       }
       delete p; 
     }
     T * p;
   };

   std::atomic<T*> cont[N];
   std::atomic<int> n{0}; 
};


#include<iostream>

struct Bar{ std::atomic<int> n{0}; std::atomic<bool> inUse{false}; };


int main() {
  auto & pool = AtomicPool<Bar,128>::pool();

  std::cout << pool.n << std::endl;

  Bar * w = nullptr;
  {
     AtomicPool<Bar,128>::Sentry sentry;
    assert(sentry.p);
    w = sentry.p;
    std::cout << pool.n << std::endl;

  }
  std::cout << pool.n << std::endl;
  {
     AtomicPool<Bar,128>::Sentry sentry;
    assert(sentry.p);
    assert(w == sentry.p);
    std::cout << pool.n << std::endl;

  }
  std::cout << pool.n << std::endl;

  std::atomic<bool> wait{true};   
  auto run = [&]() {
   while (wait);
   for (int i=0; i<1000; ++i) {
     AtomicPool<Bar,128>::Sentry sentry;
     assert(sentry.p);
     assert(!sentry.p->inUse.load(std::memory_order_acquire));
     sentry.p->inUse.store(true,std::memory_order_release);
     sentry.p->n++;
     sentry.p->inUse.store(false,std::memory_order_release);
   }
    std::cout << pool.n.load(std::memory_order_relaxed) << std::endl;
  };


  std::vector<std::thread> th;
  for (int i=0; i<10; i++) th.emplace_back(run);
  wait=false;
  for (auto & t:th) t.join();
  std::cout << pool.n << std::endl;
  int tot=0;
  for (auto const & b : pool.cont) if(b) {tot+= b.load()->n; std::cout << b.load()->n << ' ';}
  std::cout << "= " << tot << std::endl;

  return 0;
}
