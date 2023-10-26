// compile with
// c++ -O3 -pthread -fPIC -shared -std=c++23 mallocWrapper.cc -lstdc++exp -o mallocHook.so -ldl
#include <cstdint>
#include <dlfcn.h>
#include <iostream>
#include<cassert>
#include <unordered_map>
#include<map>
#include<vector>
#include <memory>

#include <mutex>
#include <thread>



#include <iostream>
#include <string>
#include <stacktrace>

namespace {


  typedef std::thread Thread;
  typedef std::vector<std::thread> ThreadGroup;
  typedef std::mutex Mutex;
  typedef std::unique_lock<std::mutex> Lock;
  // typedef std::condition_variable Condition;


  Mutex globalLock;

  std::string get_stacktrace() {
     std::string trace;
     for (auto & entry : std::stacktrace::current() ) trace += entry.description() + '#';
     return trace;
  }

  thread_local bool doRecording = true;

struct  Me {

  struct One {
    double mtot = 0;
    uint64_t mlive = 0;
    uint64_t mmax=0;
    uint64_t ntot=0;

    void add(std::size_t size) {
       mtot += size;
       mlive +=size;
       mmax = std::max( mmax,mlive);
       ntot +=1;
    }
    void sub(std::size_t size) {
     mlive -=size;
    }
  };

   enum class SortBy {none, tot, live, max, ncalls};
   using TraceMap = std::unordered_map<std::string,One>;
   using TraceVector = std::vector<std::pair<std::string,One>>;
   using Us = std::vector<std::unique_ptr<Me>>;



  Me() {
//    setenv("LD_PRELOAD","", true);
    std::cerr << "Recoding structure constructed in a thread " << std::endl;
  }

  ~Me() {
    doRecording = false;
    std::cout << "MemStat " << ntot << ' ' << mtot << ' ' << mlive << ' ' << mmax <<' ' << memMap.size() << std::endl;
    dump(std::cout,SortBy::max);
  }

  void add(void * p, std::size_t size) {
    Lock guard(lock);
    mtot += size;
    mlive +=size;
    mmax = std::max( mmax,mlive);
    ntot +=1;
    // std::cout << "m " << size << ' ' << p << std::endl;
    auto & e = calls[get_stacktrace()];
    memMap[p] = std::make_pair(size, &e);
    e.add(size);
  }

  bool sub(void * p)  {
    if (!p) return true;
    Lock guard(lock);
    // std::cout << "f " << p << std::endl;
    if (auto search = memMap.find(p); search != memMap.end()) {
     mlive -= search->second.first;
     search->second.second->sub(search->second.first);
     memMap.erase(p);
     return true;
    } 
    return false;
    // std::cout << "free not found " << p << std::endl; 
  
  }

  std::ostream & dump(std::ostream & out, SortBy sortMode) const {
     using Elem = TraceVector::value_type;
     // auto comp = [](Elem const & a, Elem const & b) {return a.first < b.first;};
     auto comp = [](Elem const & a, Elem const & b) {return a.second.mmax < b.second.mmax;};
     // if (sortMode == SortBy::live) comp = [](Elem const & a, Elem const & b) {return a.second.mlive < b.second.mlive;};
     TraceVector v;  v.reserve(calls.size());
     { 
       Lock guard(lock);
       for ( auto const & e : calls) { v.emplace_back(e.first,e.second);  std::push_heap(v.begin(), v.end(),comp);}
     }
     std::sort_heap(v.begin(), v.end(),comp);
     for ( auto const & e : v)  out << e.first << ' ' << e.second.ntot << ' ' << e.second.mtot << ' ' << e.second.mlive << ' ' << e.second.mmax << '\n';
     return out;
  }


  mutable Mutex lock;
  std::unordered_map<void*,std::pair<uint64_t,One*>> memMap; // active memory blocks 
  TraceMap calls;  // stat by stacktrace
  double mtot = 0;
  uint64_t mlive = 0;
  uint64_t mmax = 0;
  uint64_t ntot=0;

  static Us & us() {
   static Us us;
   if (us.capacity()<1024) us.reserve(1024);
   return us;
  }

  static Me & me() {
    thread_local Me * tlme = nullptr;
    if (tlme)  return *tlme; 
    
    auto l = std::make_unique<Me>();
    tlme = l.get();
    {
      Lock guard(globalLock);
      us().push_back(std::move(l));
    }
    return *tlme;
  }


  static void globalSub(void * p)  {
    std::vector<Me*> all;
    all.reserve(2*us().size());
    {
      Lock guard(globalLock);
      for (auto & e : us() ) all.push_back(e.get());
    }
    auto here = &me();
    for (auto e : all) {
     if (here == e) continue;
     if (e->sub(p)) return;
    }
    std::cout << "free not found " << p << std::endl;
  }

};



  typedef void * (*mallocSym) (std::size_t);
  typedef void (*freeSym) (void*);
  typedef void * (*aligned_allocSym)( size_t alignment, size_t size );

  typedef void * (*dlopenSym)(const char *, int);

  dlopenSym origDL = nullptr;
  mallocSym origM = nullptr;
  freeSym origF = nullptr;
  aligned_allocSym origA = nullptr;


  struct Banner {
    Banner() {
      printf("malloc wrapper loading\n");
      fflush(stdout);
      // origM = (mallocSym)dlsym(RTLD_NEXT,"malloc");
      // origF = (freeSym)dlsym(RTLD_NEXT,"free");
    }
  };

  Banner banner;

}

// extern void *__libc_malloc(size_t size);
// extern void __libc_free(void *);


extern "C" 
{


void *dlopen(const char *filename, int flags) {
  if (!origDL) origDL = (dlopenSym)dlsym(RTLD_NEXT,"dlopen");
  doRecording = false;
  auto p  = origDL(filename,flags);
  doRecording = true;
  return p;
}


void *malloc(std::size_t size) {
  if (!origM) origM = (mallocSym)dlsym(RTLD_NEXT,"malloc");
  assert(origM);
  auto p  = origM(size); 
  if (doRecording) {
    doRecording = false;
    Me::me().add(p, size);
    doRecording = true;
  }
  return p;
}

void *aligned_alloc( size_t alignment, size_t size ) {
  if (!origA) origA = (aligned_allocSym)dlsym(RTLD_NEXT,"aligned_alloc");
  assert(origA);
  auto p  = origA(alignment, size);
  if (doRecording) {
    doRecording = false;
    Me::me().add(p, size);
    doRecording = true;
  }
  return p;

}


void free(void *ptr) {
  if(!origF) origF = (freeSym)dlsym(RTLD_NEXT,"free");
  assert(origF);
  if (doRecording) {
    doRecording = false;
    Me::me().sub(ptr);
    doRecording = true;
  }
  origF(ptr);
}


}

void Hello() {
  std::cout << "Hello" << std::endl;
  doRecording = false;
  Me::me().dump(std::cout, Me::SortBy::max);
  doRecording = true;
}

