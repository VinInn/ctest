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


#include <iostream>
#include <string>
#include <stacktrace>

namespace {

  std::string get_stacktrace() {
     std::string trace;
     for (auto & entry : std::stacktrace::current() ) trace += entry.description() + '#';
     return trace;
  }

  thread_local bool notRecording = true;

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




  Me() {
//    setenv("LD_PRELOAD","", true);
    std::cerr << "Recoding structure constructed in a thread " << std::endl;
  }

  ~Me() {
    notRecording = false;
    std::cout << "MemStat " << ntot << ' ' << mtot << ' ' << mlive << ' ' << mmax <<' ' << memMap.size() << std::endl;
    dump(std::cout,SortBy::max);
  }

  void add(void * p, std::size_t size) {
    mtot += size;
    mlive +=size;
    mmax = std::max( mmax,mlive);
    ntot +=1;
    // std::cout << "m " << size << ' ' << p << std::endl;
    auto & e = calls[get_stacktrace()];
    memMap[p] = std::make_pair(size, &e);
    e.add(size);
  }

  void sub(void * p)  {
    // std::cout << "f " << p << std::endl;
    if (auto search = memMap.find(p); search != memMap.end()) {
     mlive -= search->second.first;
     search->second.second->sub(search->second.first);
     memMap.erase(p);
    } else if (p) { std::cout << "free not found " << p << std::endl; }
  }

  std::ostream & dump(std::ostream & out, SortBy sortMode) const {
     using Elem = TraceVector::value_type;
     // auto comp = [](Elem const & a, Elem const & b) {return a.first < b.first;};
     auto comp = [](Elem const & a, Elem const & b) {return a.second.mmax < b.second.mmax;};
     // if (sortMode == SortBy::live) comp = [](Elem const & a, Elem const & b) {return a.second.mlive < b.second.mlive;};
     TraceVector v;  v.reserve(calls.size());
     for ( auto const & e : calls) { v.emplace_back(e.first,e.second);  std::push_heap(v.begin(), v.end(),comp);}
     std::sort_heap(v.begin(), v.end(),comp);
     for ( auto const & e : v)  out << e.first << ' ' << e.second.ntot << ' ' << e.second.mtot << ' ' << e.second.mlive << ' ' << e.second.mmax << '\n';
     return out;
  }

  std::unordered_map<void*,std::pair<uint64_t,One*>> memMap; // active memory blocks 
  TraceMap calls;  // stat by stacktrace
  double mtot = 0;
  uint64_t mlive = 0;
  uint64_t mmax = 0;
  uint64_t ntot=0;

  static Me & me() {
    thread_local auto me = std::make_unique<Me>();
    return *me;
  }

};



  typedef void * (*mallocSym) (std::size_t);
  typedef void (*freeSym) (void*);
  mallocSym origM = nullptr;
  freeSym origF = nullptr;

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


void *malloc(std::size_t size) {
  if (!origM) origM = (mallocSym)dlsym(RTLD_NEXT,"malloc");
  assert(origM);
  auto p  = origM(size); 
  if (notRecording) {
    notRecording = false;
    Me::me().add(p, size);
    notRecording = true;
  }
  return p;
}




void free(void *ptr) {
  if(!origF) origF = (freeSym)dlsym(RTLD_NEXT,"free");
  assert(origF);
  if (notRecording) {
    notRecording = false;
    Me::me().sub(ptr);
    notRecording = true;
  }
  origF(ptr);
}


}

void Hello() {
  std::cout << "Hello" << std::endl;
  notRecording = false;
  Me::me().dump(std::cout, Me::SortBy::max);
  notRecording = true;
}

