#include<atomic>
#include<cstdint>
#include<algorithm>
#include<cassert>

template<typename T, int N>
struct AtomicPool {
   using Self = AtomicPool<T,N>;
   static Self & pool() {
     static Self me;
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
       int n = std::min(N,pool.n.load());
       p = nullptr;
       for (int i=0; i<n; ++i) {
          p = pool.cont[i].exchange(nullptr);
         if (p) return;
       }
       p = new T();
     }
     ~Sentry() {
       auto & pool = Self::pool();
       int n = std::min(N,pool.n.load());
       for (int i=0; i<n; ++i) {
         T * exp = nullptr;
         if (pool.cont[i].compare_exchange_strong(exp,p)) return;
       }
       n = pool.n++;
       if (n<N) {
         T * exp = nullptr;
       if (pool.cont[n].compare_exchange_strong(exp,p)) return;
       }
       delete p; 
     }
     T * p;
   };

   std::atomic<T*> cont[N];
   std::atomic<int> n=0; 
};


#include<iostream>

struct Bar{};


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



  return 0;
}
