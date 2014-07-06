#include "RealTime.h"

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




struct ImageProducer {
  typedef unsigned char value_type;
  enum { imageSize=10000000};
  ImageProducer() : q(5), done(0) {}

  void operator()() {
    while(true) {
      value_type * image = new value_type[imageSize];
      for (int i=0; i<imageSize; ++i)
	image[i] = 255&rand();
      q.push(image);
      done++;
    }
  }
  Queue<value_type*> q;
  int done;
};

struct Worker {
  typedef ImageProducer::value_type value_type;

  Worker(Queue<value_type*>& iq) : q(iq), hist(256,0){}
  
  void operator()() {
    waitStart(); // barrier
    while(active) {
      value_type * i;
      while(q.pop(i)) {
	value_type * e = i+4000;
	for (value_type * k=i; k!=e; k++)
	  ++hist[*k];
      }
    }
  }
  void zero() {
    std::fill(hist.begin(),hist.end(),0);
  }

  static void waitStart() {
    __sync_add_and_fetch(&start,-1);
    do{}while(start);
  }

  Queue<value_type*>& q;
  std::vector<int> hist;
  static bool active;
  static volatile long start;
};

bool Worker::active=true;
volatile long Worker::start=0;


int main(int argc, char * argv[]) {


  int NUMTHREADS=10;
  if (argc>1) NUMTHREADS=atoi(argv[1]);

  ImageProducer producer;
  Thread p1(std::ref(producer));
  p1.detach();


  Queue<Worker::value_type*> q(30);
    
  size_t stride = 4000; // shall match L1 cache
  
   __sync_lock_test_and_set(&Worker::start,NUMTHREADS+1);
  ThreadGroup threads;
  threads.reserve(NUMTHREADS);
  std::vector<Worker> workers(NUMTHREADS, Worker(q));
  for (int i=0; i<NUMTHREADS; ++i) {
    threads.push_back(Thread(std::ref(workers[i])));
  }

  // we shall wait for all threads to be ready...
  // (for timing)
  do{}while(Worker::start!=1);
   // start worker
  __sync_add_and_fetch(&Worker::start,-1);

  long long mapTime=0;
  long long reduceTime=0;
  for (int l=0; l<10;++l)
  {
    // reset queue;
    //q.reset();
 
    if ( producer.q.empty() ) std::cout << "producer empty" << std::endl;
    if ( producer.q.full() ) std::cout << "producer full" << std::endl;
    ImageProducer::value_type * image;
    producer.q.pop(image);
    //map
    long long st = rdtsc();
    ImageProducer::value_type * curr = image;
    ImageProducer::value_type * end = image+ ImageProducer::imageSize;
    while(curr<end) {
      // std::cout << curr-image << " " << (int)(*curr) << std::endl;
      q.push(curr);
      curr+=stride;
    }  
    
    // barrier
    do{} while (!q.empty());
    mapTime+= rdtsc()-st;

    // reduce
    std::vector<int> hist(256);
    st = rdtsc();
    for (int i=0; i!=NUMTHREADS;++i)
      for (int j=0; j!=256;++j) 
	hist[j]+= workers[i].hist[j];
    reduceTime+= rdtsc()-st;
    
    for (int i=0; i!=NUMTHREADS;++i) {
      std::cout << "thread "<< i << " : ";
      for (int j=0; j!=256;++j)
	std::cout << workers[i].hist[j] << " ,";
      std::cout << std::endl;
    }
   
    std::cout << "\nTotal " << l << std::endl;
    for (int j=0; j!=256;++j) 
      std::cout << hist[j] << " ,";
    std::cout << std::endl;

    delete [] image;

    // prepare new loop (actually part of reduce step)
    for (int i=0; i<NUMTHREADS; ++i) workers[i].zero();
  }

  Worker::active=false;
  q.drain();
  std::for_each(threads.begin(),threads.end(), 
		std::bind(&Thread::join,std::placeholders::_1));
  

  std::cout << "map time " << double(mapTime)/1000. << std::endl;
  std::cout << "reduce time " << double(reduceTime)/1000. << std::endl;
     
  return 0;

  }
