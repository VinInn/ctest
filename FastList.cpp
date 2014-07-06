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



template<typename T>
class SharedList {
public:
  typedef T value_type;
  
  struct Node {
    Node() : next(0){}
    volatile Node * next;
    value_type value;
  }; //  __attribute__ ((__aligned__(CACHE_LINE_SIZE)));

  typedef Node volatile * pointer;
  typedef Node volatile const * const_pointer;

  SharedList() : head(), collisions(100,0){}

  pointer begin() { return head.next;}
  const_pointer begin() const { return head.next;}
  const_pointer end() const { return 0;}

  // insert AFTER p
  pointer insert(pointer p, value_type const & value) {
    Node * me = new Node;
    me->value=value;
    // if (head==0 && __sync_bool_compare_and_swap(&head,0,me)) return me;
    if (p==0) p=&head;
    while (true) {
      me->next = p->next;
      if (__sync_bool_compare_and_swap(&(p->next),me->next,me)) break;
      // __sync_add_and_fetch(&(collisions[int(value)]),1);
      ++collisions[int(value)];
    }
    return me;
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
   p = p->next;
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
  
  //  dump(list);
  
  for (int i=0; i!=NUMTHREADS;++i) 
    std::cout << list.collisions[i] << " ,";
  std::cout << std::endl;
  
  std::cout << "time inserting " << tot <<std::endl;
	    
  return 0;


}
