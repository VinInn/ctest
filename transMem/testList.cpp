#include <vector>
#include <cmath>
#include <limits>
#include <queue>
#include <mutex>
#include <thread>
#include <functional>
#include <algorithm>
#include <immintrin.h>

// convert gcc to c0xx
#define thread_local __thread;

typedef std::thread Thread;
typedef std::vector<std::thread> ThreadGroup;
typedef std::mutex Mutex;
typedef std::lock_guard<std::mutex> Lock;
// typedef std::unique_lock<std::mutex> Lock;
// typedef std::condition_variable Condition;

#ifdef NOTRANS
#define __transaction_atomic
#endif

#ifdef HTMEM
#define __transaction_atomic unsigned int status; if ((status = _xbegin ()) == _XBEGIN_STARTED)
#else
void _xend (){}
#endif

class Set {
public:

  using Value=float;

  struct Node {
    Node() : val(0), next(nullptr) {} 
    Node(Value ival, Node * inext) : val(ival), next(inext) {} 
    Value val;
    Node *next;
  };

 
  Node head;

  bool contains(Value val) const {
    bool result=false;
    Node const * prev, *next;

    __transaction_atomic {
      prev = &head;
      next = prev->next;
      while (next && next->val < val) {
	prev = next;
	next = prev->next;
      }
      result = (next && next->val == val);
      _xend ();
    }
    return result;
  }
  
  bool add(Value val) {
    bool result=false;
    Node *prev, *next;
    
    __transaction_atomic {
      prev = &head;
      next = prev->next;
      while (next && next->val < val) {
	prev = next;
	next = prev->next;
      }
      result = (nullptr==next || next->val != val);
      if (result) {
	prev->next = new Node(val, next);
      }
      _xend ();
    }
    return result;
  }


  bool remove(Value val) {
    bool result=false;
    Node *prev, *next;
    
    __transaction_atomic {
      prev = &head;
      next = prev->next;
      while (next && next->val < val) {
	prev = next;
	next = prev->next;
      }
      result = (nullptr!=next && next->val == val);
      if (result) {
	prev->next =  next->next;
	delete next;
      }
      _xend ();
    }
    return result;
  }

  int size() const {
    int res=0;
    Node const *prev, *next;
    
    __transaction_atomic {
      prev = &head;
      next = prev->next;
      while (next) {
	prev = next;
	next = prev->next;
	++res;
      }
      _xend ();
    }
    return res;
  }

};

#include<iostream>
// control cout....
Mutex outLock;

Set theSet;


bool stop=false;
void stopper() {
  {
    Lock a(outLock);
    std::cout << "I'm the stopper" << std::endl;
  }

  //  char s;
  //std::cin >> s;
  while (nullptr==theSet.head.next || theSet.head.next->val > -1.e6) {
    std::chrono::milliseconds dura( 2000 );
    std::this_thread::sleep_for( dura );
  }

  stop=true;
}


void act() {

  int me=0;

  __transaction_atomic {
    // not really needed atomic will be enough
    static int nt=0;
    me = nt++;
  }  

  {
    Lock a(outLock);
    std::cout << "starting " <<  me << std::endl;
  }
  int wrong=0;

  float n=me*100+0.01*me;
  float oldN=0;
  bool back = (0==(me&1));
  if (!back) n = -n;
  while(!stop) {
    
    n += back ? -1 : 1;
    theSet.add(n);
    if (!theSet.contains(n)) wrong++;
    theSet.remove(oldN);
    oldN=n;
  }

  {
    Lock a(outLock);
    std::cout << me << " " << wrong << std::endl;
  }

}

int main() {
  Set aset;
  std::cout << aset.size() << std::endl;

  if ( aset.add(1)) std::cout << "oK" << std::endl;
  if ( aset.add(1)) std::cout << "NoK" << std::endl;


  if ( aset.contains(1)) std::cout << "oK" << std::endl;
  if ( aset.contains(-1)) std::cout << "noK" << std::endl;
  if ( aset.contains(10)) std::cout << "noK" << std::endl;
  std::cout << aset.size() << std::endl;
  if ( aset.remove(1)) std::cout << "oK" << std::endl;
  std::cout << aset.size() << std::endl;
  if ( aset.remove(1)) std::cout << "NoK" << std::endl;
  std::cout << aset.size() << std::endl;
  if ( aset.remove(-1)) std::cout << "noK" << std::endl;
  if ( aset.remove(10)) std::cout << "noK" << std::endl;
  std::cout << aset.size() << std::endl;

  const int NUMTHREADS=10;
  ThreadGroup threads;
  threads.reserve(NUMTHREADS);
  for (int i=0; i<NUMTHREADS; ++i) {
    threads.push_back(Thread(act));
  }

  Thread s(stopper);
  s.detach();

  std::for_each(threads.begin(),threads.end(), 
	       std::bind(&Thread::join,std::placeholders::_1));
  

  std::cout << theSet.size() << std::endl;
  auto next = &theSet.head;
  while (next) {std::cout << next->val << " "; next=next->next;}
  std::cout << std::endl;

  return 0;

}
