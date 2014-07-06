#ifndef CircularBuffer_H
#define CircularBuffer_H

#include<vector>
#include<atomic>

#include <thread>

#define CACHE_LINE_SIZE  64 // 64 byte cache line on x86 and x86-64


template<typename T>
class CircularBuffer {
public:
  explicit CircularBuffer(size_t maxsize) : 
  head(maxsize-1), tail(maxsize-1), last(maxsize-1), 
  container(maxsize),  m_pushLock(false), drained(false) {}
  
  typedef T AT __attribute__ ((__aligned__(CACHE_LINE_SIZE)));

  void waitFull() const{
    // spinlock
    while(full()) {
      nanosleep(0,0);
      // std::this_thread::yield();
      // std::this_thread::sleep_for(0);
    } 
  }
  
  bool waitEmpty() const{
    // spinlock
    while(empty()&&(!drained))
    {
      nanosleep(0,0);
      //std::this_thread::sleep_for(0);
      // std::this_thread::yield();
    } 
    return empty()&&drained;
  }
  
  // only one thread can push
  bool push(T && t, bool wait=true) {
    while (true) {
      if (wait) waitFull();
      else if (full()) return false;
      size_t cur=head;
      container[cur] = t; // shall be done first to avoid popping wrong value
      // does not work: if fails state of head is unknown....
      if (std::atomic_compare_exchange_weak(&head,&cur,cur==0 ? last : cur-1 )) {
	// container[cur] = t; // too late pop already occured!
	break;
      }
    }
    return true;
  }
  
  // N threads can pop
  bool pop(T&t, bool wait=true) {
    while (true) {
      if (wait) {
	if(waitEmpty()) return false; // include a signal to drain and terminate
      }
      else if (empty()) return false;
      size_t cur=tail;
      if (cur==head) continue;
      // t = std::move(container[cur]);
      t = container[cur];
      if (std::atomic_compare_exchange_weak(&tail,&cur,cur==0 ?last : cur-1)) break;
    }
    return true;
  }
  
  std::atomic<bool> & pushLock() { return m_pushLock;}
  bool tryLock() {
    if (pushLock()) return false;
    bool ok = false;
    return std::atomic_compare_exchange_strong(&m_pushLock,&ok, true);
  }
  void releaseLock() { m_pushLock=false; }

  size_t size() const { return std::max(std::ptrdiff_t(tail)-std::ptrdiff_t(head),std::ptrdiff_t(head)-std::ptrdiff_t(tail)); }
  bool halfEmpty() const { return 2*size() < last;}


  bool full() const { return (head==0 && tail==last) 
      || (tail==head-1);
  }
  bool empty() const { return head==tail;}
  
  void drain() { drained=true;}
  bool draining() const { return drained;}
  void reset() { drained=false; head=tail=last;}
  
  // circular buffer
  std::atomic<size_t> head;
  std::atomic<size_t> tail;
  size_t last;
  std::vector<AT> container;
  std::atomic<bool> m_pushLock;

  bool drained;
};


#endif




