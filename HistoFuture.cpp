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

#include "future.h"

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

#include "SharedQueue.h"

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
  typedef std::pair<value_type*,value_type*> range;

  Worker() : hist(256,0){}
  
  std::unique_future<range> input() {
    return m_input.get_future();
  }

   std::unique_future<std::vector<int>& > output() const {
     return m_output.get_future();
  }

  void set(range r) {
    m_input.set_value(r);
  }

  std::vector<int> const & get() const {
    output().get();
    std::promise< std::vector<int>& >().swap(m_output);
    return hist;
  }

  void reset() { 
    std::promise<range>().swap(m_input); 
    zero();
  }

  void operator()() {
    waitStart(); // barrier
    while(active) {
      try {
	range r = input().get();
	reset();
	for (value_type * k=r.first; k!=r.second; ++k)
	  ++hist[*k];
	m_output.set_value(hist);
      } catch(...) {
	break;
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

  
  mutable std::promise<range> m_input;
  mutable std::promise< std::vector<int>& > m_output;
  std::vector<int> hist;
  static bool active;
  static volatile long start;
};

bool Worker::active=true;
volatile long Worker::start=0;

int main(int argc, char * argv[]) {


  int NUMTHREADS=8;
  if (argc>1) NUMTHREADS=atoi(argv[1]);



  ImageProducer producer;
  Thread p1(std::ref(producer));
  p1.detach();


   
  
   __sync_lock_test_and_set(&Worker::start,NUMTHREADS+1);
  ThreadGroup threads;
  threads.reserve(NUMTHREADS);
  std::vector<std::shared_ptr<Worker> > workers(NUMTHREADS);
  for (int i=0; i<NUMTHREADS; ++i) {
    workers[i].reset(new Worker);
    threads.push_back(Thread(std::ref(*workers[i])));
  }

  // we shall wait for all threads to be ready (just for timing)...
  do{}while(Worker::start!=1);
  // start worker
  __sync_add_and_fetch(&Worker::start,-1);

  long long mapTime=0;
  long long reduceTime=0;
  for (int l=0; l<10;++l)
  {
     if ( producer.q.empty() ) std::cout << "producer empty" << std::endl;
    if ( producer.q.full() ) std::cout << "producer full" << std::endl;
    ImageProducer::value_type * image;
    producer.q.pop(image);
    //map
    long long st = rdtsc();
    ImageProducer::value_type * curr = image;
    size_t stride = ImageProducer::imageSize/NUMTHREADS;
 
    ImageProducer::value_type * end = image+ ImageProducer::imageSize;
    for (int i=0; i!=NUMTHREADS;++i) {
      workers[i]->set(Worker::range(curr, std::min(curr+stride,end)));
      curr+=stride;
    }  
    
    // barrier (just for timing)
    for (int i=0; i!=NUMTHREADS;++i)
      workers[i]->get();
    mapTime+= rdtsc()-st;

    // reduce
    std::vector<int> hist(256);
    st = rdtsc();
    for (int i=0; i!=NUMTHREADS;++i)
      for (int j=0; j!=256;++j) 
	hist[j]+= workers[i]->hist[j];
    reduceTime+= rdtsc()-st;
    
    for (int i=0; i!=NUMTHREADS;++i) {
      std::cout << "thread "<< i << " : ";
      for (int j=0; j!=256;++j)
	std::cout << workers[i]->hist[j] << " ,";
      std::cout << std::endl;
    }
   
    std::cout << "\nTotal " << l << std::endl;
    for (int j=0; j!=256;++j) 
      std::cout << hist[j] << " ,";
    std::cout << std::endl;

    delete [] image;

    // prepare new loop
  
    // for (int i=0; i<NUMTHREADS; ++i) workers[i].zero();
  }

  Worker::active=false;
  for (int i=0; i!=NUMTHREADS;++i)
    workers[i]->set(Worker::range());

  std::for_each(threads.begin(),threads.end(), 
		std::bind(&Thread::join,std::placeholders::_1));
  

  std::cout << "map time " << double(mapTime)/1000. << std::endl;
  std::cout << "reduce time " << double(reduceTime)/1000. << std::endl;
     
  return 0;

  }
