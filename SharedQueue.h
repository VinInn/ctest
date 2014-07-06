#ifndef SharedQueue_H
#define SharedQueue_H

#include<vector>

#include <thread>
#include <functional>
#include <algorithm>
#include<cmath>

#include<iostream>

// convert gcc to c0xx
#define thread_local __thread;

typedef std::thread Thread;
typedef std::vector<std::thread> ThreadGroup;
typedef std::mutex Mutex;
typedef std::unique_lock<std::mutex> Guard;
typedef std::condition_variable Condition;

#define CACHE_LINE_SIZE  64 // 64 byte cache line on x86 and x86-64

/*
typedef<typename T>
struct Padded {
  Padded(T& const i) : t(i){}
  Padded(T&&i) : t(std::move(i)){}
  T t;
  char pad[CACHE_LINE_SIZE-sizeof(T)%CACHE_LINE_SIZE];
};
*/

template<typename T>
class Queue {
public:
  explicit Queue(size_t maxsize) : head(maxsize-1), tail(head), last(maxsize-1), 
    container(maxsize), drained(false) {}
  
  typedef T AT __attribute__ ((__aligned__(CACHE_LINE_SIZE)));

  void waitFull() const{
    // spinlock
    while(full()) {
      nanosleep(0,0);
      //      std::this_thread::yield();
      // std::this_thread::sleep_for(0);
    } 
  }
  
  bool waitEmpty() const{
    // spinlock
    while(empty()&&(!drained))
    {
      nanosleep(0,0);
      //std::this_thread::sleep_for(0);
       //     std::this_thread::yield();
    } 
    return empty()&&drained;
  }
  
  // only one thread can push
  void push(T const & t) {
    while (true) {
      waitFull();
      volatile size_t cur=head;
      container[cur] = t; // shall be done first to avoid popping wrong value
      // does not work: if fails state of head is unknown....
      if (__sync_bool_compare_and_swap(&head,cur,cur==0 ? last : cur-1 )) {
	// container[cur] = t; // too late pop already occured!
	break;
      }
    }
  }
  
  // N threads can pop
  bool pop(T&t) {
    while (true) {
      if(waitEmpty()) return false; // include a signal to drain and terminate
      volatile size_t cur=tail;
      if (cur==head) continue;
      t = container[cur];
      if (__sync_bool_compare_and_swap(&tail,cur,cur==0 ?last : cur-1)) break;
    }
    return true;
  }
  
  bool full() const { return (head==0 && tail==last) 
      || (tail==head-1);
  }
  bool empty() const { return head==tail;}
  
  void drain() { drained=true;}
  void reset() { drained=false; head=tail=last;}
  
  // circular buffer
  volatile size_t head;
  volatile size_t tail;
  size_t last;
  std::vector<AT> container;
  bool drained;
};


#endif




