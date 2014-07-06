#include "RealTime.h"

#include<iostream>
#include <thread>
#include <cstdatomic>
#include <functional>
#include <algorithm>
#include<vector>
#include<sstream>

// convert gcc to c0xx
#define thread_local __thread

typedef std::thread Thread;
typedef std::vector<std::thread> ThreadGroup;
typedef std::mutex Mutex;
typedef std::unique_lock<std::mutex> Guard;
typedef std::condition_variable Condition;

#define CACHE_LINE_SIZE  64 // 64 byte cache line on x86 and x86-64


long long threadId() {
  std::stringstream ss;
  ss << std::this_thread::get_id();
  long long id;
  ss >> id;
  return id;
}


// only 64 bit!
const long tombMask= 0x80000000;
const long pointerMask= 0x7fffffff;

inline bool tombStone(void const * const volatile p) {
  return ((long)(p)&tombMask)!=0;
}

template<typename T>
inline void setTombStone(T * volatile & p) {
  __sync_or_and_fetch(&p,tombMask);
}

template<typename T>
inline T * asPointer(T * volatile p) {
  return (T*)((long)(p)&pointerMask);
}



template<typename T>
class SharedList {
public:
  typedef T value_type;
  
  struct Node {
    Node() : m_next(0){}
    bool valid() const { return !tombStone(m_next); }
    void markInvalid() { setTombStone(m_next);}
    bool last() const { return 0==asPointer(m_next);}
    Node * volatile next() { return asPointer(m_next); }
    Node * volatile nextValid() {
      return nv();
    }
    Node const * volatile nextValid() const {
      return const_cast<Node*>(this)->nv();
    }
    Node * volatile nv() {
      if (valid()) return m_next;
      if (last()) return 0;
      return next()->nextValid();
    }
    Node * volatile m_next;
    value_type value;
  }; //  __attribute__ ((__aligned__(CACHE_LINE_SIZE)));
  
  typedef Node *  volatile pointer;
  typedef Node const *  volatile const_pointer;
  
  SharedList() : head(), collisions(100,0){
    // to remove init from timing 
    alloc();
  }
  
  pointer begin() { return head.nextValid();}
  const_pointer begin() const { return head.nextValid();}
  const_pointer end() const { return 0;}

   Node * alloc()  {
     // return new Node;
     
     static thread_local std::vector<Node> * pool=0;
     static thread_local int last=0;
     if (0==last) pool = new std::vector<Node>(150000);
     return &(*pool)[last++];
     
    }
   

  // insert AFTER p
  pointer insert(pointer p, value_type const & value) {
    Node * me = alloc();
    me->value=value;
    // if (head==0 && __sync_bool_compare_and_swap(&head,0,me)) return me;
    if (p==0) p=&head;
    while (true) {
      me->m_next = p->m_next;
      if (__sync_bool_compare_and_swap(&(p->m_next),me->m_next,me)) break;
      // __sync_add_and_fetch(&(collisions[int(value)]),1);
      ++collisions[int(value)];
    }
    return me;
  }


  // "remove"
  pointer remove(pointer p) {
    // never delete on the fly! just mark
    p->markInvalid();
    return p->nextValid();
  }

private:

  Node head;
public:
typedef int Aint; // __attribute__ ((__aligned__(CACHE_LINE_SIZE)));

  std::vector<Aint> collisions;
};

template<typename T>
void dump(SharedList<T> const & list) {
 size_t size=0;
 typename SharedList<T>::const_pointer p= list.begin();
 while(p) {
   ++size;
   std::cout << size << ": " << (*p).value << std::endl;
   p = p->nextValid();
 }

}

struct Inserter {
  static volatile long start;
  static volatile long stop;

  Inserter(SharedList<float> & ilist, float iid) : list(ilist),id(iid){}

  void wait() {
    __sync_add_and_fetch(&start,-1);
    while(start) nanosleep(0,0);
  }
  void operator()() {
    // wait everybody is ready;
    wait();
    // strict collision (or no collision)....
    SharedList<float>::pointer p=list.begin();
    for (int i=0; i<100000; ++i) {
      list.insert(list.begin(),id);
      // p = list.insert(p,id);
    }
    __sync_add_and_fetch(&stop,-1);

  }
  SharedList<float> & list;
  float id;
};


volatile long Inserter::start=0;
volatile long Inserter::stop=0;


int main() {
  
  const int NUMTHREADS=10;
  Inserter::start=NUMTHREADS+1;
  Inserter::stop=NUMTHREADS;
  
  SharedList<float> list;
  
  ThreadGroup threads;
  threads.reserve(NUMTHREADS);
  for (int i=0; i<NUMTHREADS; ++i) {
    threads.push_back(Thread(Inserter(list,float(i))));
  }
  
  long long st =  rdtsc();
  __sync_add_and_fetch(&Inserter::start,-1);
  do{}while(Inserter::stop);
  long long tot = rdtsc()-st;
  
  std::for_each(threads.begin(),threads.end(), 
		std::bind(&Thread::join,std::placeholders::_1));
  
  // dump(list);
  
  for (int i=0; i!=NUMTHREADS;++i) 
    std::cout << list.collisions[i] << " ,";
  std::cout << std::endl;
  
  std::cout << "time inserting " << tot <<std::endl;
	    
  return 0;


}
