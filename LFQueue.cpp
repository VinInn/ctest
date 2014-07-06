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



template<typename T>
class Queue {
public:
  explicit Queue(size_t maxsize) : head(maxsize-1), tail(head), last(maxsize-1), 
    container(maxsize), drained(false) {}
  
  void waitFull() const{
    // spinlock
    do{} while(full());
  }
  
  bool waitEmpty() const{
    // spinlock
    do{} while(empty()&&(!drained));
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
  
  // circular buffer
  volatile size_t head;
  volatile size_t tail;
  size_t last;
  std::vector<T> container;
  bool drained;
};



struct Worker {

  Worker(Queue<int>& iq) : q(iq), hist(100){}
  
  void operator()() {
    int i;
    while(q.pop(i)) {
      ++hist[i];
    }
    
  }
  
  Queue<int>& q;
  std::vector<int> hist;

};


int main() {
  
 const int NUMTHREADS=10;


  Queue<int> q(30);
  
  
  ThreadGroup threads;
  threads.reserve(NUMTHREADS);
  std::vector<Worker> workers(NUMTHREADS, Worker(q));
  for (int i=0; i<NUMTHREADS; ++i) {
    threads.push_back(Thread(std::ref(workers[i])));
  }

  for (int i=0; i<10000; i++) 
    q.push(i%100);
  
  q.drain();
  std::for_each(threads.begin(),threads.end(), 
		std::bind(&Thread::join,std::placeholders::_1));
  
 
  std::vector<int> hist(100);
  for (int i=0; i!=NUMTHREADS;++i) {
    std::cout << "thread "<< i << " : ";
    for (int j=0; j!=100;++j) {
      hist[j]+= workers[i].hist[j];
      std::cout << workers[i].hist[j] << " ,";
    }
    std::cout << std::endl;
  }

  std::cout << "\nTotal " << std::endl;
  for (int j=0; j!=100;++j) 
    std::cout << hist[j] << " ,";
  std::cout << std::endl;
    
  return 0;

  }
