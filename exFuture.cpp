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

template<typename F>
std::unique_future<typename std::result_of<F()>::type> spawn_task(F f) {
  typedef typename std::result_of<F()>::type result_type;
  std::packaged_task<result_type()> task(std::move(f));
  std::unique_future<result_type> res(task.get_future());
  std::thread t(std::move(task));
  t.detach();
  return res;
}



struct AnError{};

struct Producer {

  Producer(bool ir=false) : error(ir){}

  std::unique_future<int> get() {
    return m_promise.get_future();
  }

  void operator()() {
    try {
      if (error) throw AnError();
      int res = 2;
      m_promise.set_value(res);
    }
    catch (...) {
      m_promise.set_exception(std::current_exception());
    }
  }

  void reset() { std::promise<int>().swap(m_promise);}

  std::promise<int> m_promise;
  bool error;
};

struct Client {



  void operator()() {
    try {
      std::unique_future<int> input = prod->get();
      int i = input.get();
      std::cout << "got " << i << std::endl;
    } catch (AnError const & ce) {
      std::cout << "got a known error" << std::endl;
    } catch (...) {
      std::cout << "got an error" << std::endl;
    }
  }

  Producer * prod;
};


struct Accumulator {

  Accumulator(){}
  Accumulator(const Accumulator&) = delete;
  Accumulator(Accumulator&& rh)  {} // : m_promise(std::move(rh.m_promise)){}
  Accumulator & operator=(const Accumulator&) = delete;
  Accumulator & operator=(Accumulator&& rh) {
    // m_promise.swap(rh.m_promise);
    return * this;
  }

  int get() {
    return m_promise.get_future().get();
  }
  
  void operator()() {
    for (int i=0; i!=inputs.size(); ++i) {
      product+= inputs[i]->get();
    }
    m_promise.set_value(product);
  }
  
  
  std::vector<std::shared_ptr<Accumulator> > inputs;
  
  int product;
  
  void reset() { std::promise<int>().swap(m_promise);}
  
  std::promise<int> m_promise;
  
};


int main() {


  {  
    Producer prod;
    Client client;
    client.prod = &prod;
    
    
    Thread c(std::ref(client));
    std::cout << "now start prod" << std::endl;
    Thread p(std::ref(prod));
    
    c.join();
    p.join();
  }

  {  
    Producer prod(true);
    Client client;
    client.prod = &prod;
    
    
    Thread c(std::ref(client));
    std::cout << "now start prod" << std::endl;
    Thread p(std::ref(prod));
    
    c.join();
    p.join();
  }
  
  int NTHREADS = 8;
  std::vector<std::shared_ptr<Accumulator> >workers(NTHREADS);
  // workers.reserve(NTHREADS);
  for (int i=0; i<NTHREADS; ++i) 
    workers[i].reset(new Accumulator());
  //    workers.push_back(Accumulator());
  
  int n = NTHREADS;
  while (n!=1) {
    n = n/2;
    for (int i=0; i!=n; i++) {
      int j = n+i;
      workers[i]->inputs.push_back(workers[j]);
    }  
  }
  
  for (int i=0; i<NTHREADS; ++i)
    std::cout << workers[i]->inputs.size() << ", ";
  std::cout << std::endl;

  ThreadGroup threads;
  threads.reserve(NTHREADS);
  for (int i=0; i<NTHREADS; ++i) {
    workers[i]->product=i;
    threads.push_back(Thread(std::ref(*workers[i])));
  }
  
  std::for_each(threads.begin(),threads.end(), 
		std::bind(&Thread::join,std::placeholders::_1));
  
  for (int i=0; i<NTHREADS; ++i)
    std::cout << workers[i]->product << ", ";
  std::cout << std::endl;
  
  return 0;
  
}


