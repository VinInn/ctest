#include <iostream>
#include <thread>
#include <functional>

#include<vector>
#include <algorithm>

#include <exception>
#include <sstream>

// convert gcc to c0xx
#define thread_local __thread

typedef std::thread Thread;
typedef std::vector<std::thread> ThreadGroup;
typedef std::mutex Mutex;
// typedef std::lock_guard<std::mutex> Lock;
typedef std::unique_lock<std::mutex> Lock;
typedef std::condition_variable Condition;


long long threadId() {
  std::stringstream ss;
  ss << std::this_thread::get_id();
  long long id;
  ss >> id;
  return id; 
}


namespace global {
  // control cout....
  Mutex coutLock;
}

void hi() {
  std::cout << "Hello World from "
	    << std::this_thread::get_id() << std::endl;
}

thread_local long long id=0;

class Hi {
public:
//  static thread_local long long id;
  static long volatile start;
  Hi() : j(0) {}
  explicit Hi(int i) : j(i){}
 
  void wait() {
    __sync_add_and_fetch(&start,-1);
    do{} while (start>0);	
  }

  int getJ() const { return j;}
  
  void operator()() {
    id = threadId();
    wait();
    // ++j;
    j = getJ()+1;
    std::cout << "Hello " << "World " << "from "
	      << id // std::this_thread::get_id() 
              << " where j is " << j 
              << " start is " << Hi::start << std::endl;    
  }
  volatile int j;  
};

long volatile Hi::start=1;

void hello() {
  
  
  std::thread t1(hi);
  std::thread t2(Hi(3)); // pass copy of local object
  hi();

  t1.join();
  t2.join();

  Hi::start=3;	
  Hi oneHi; // create local object
  std::thread t3(std::ref(oneHi)); // pass reference to oneHi
  std::thread t4(std::ref(oneHi));
  std::cout << "start is " << Hi::start << std::endl;
  oneHi();
  std::cout << "start is " << Hi::start << std::endl;
  
  t3.join();
  t4.join();
  std::cout << "j is " << oneHi.j << std::endl;

}


class K {
public:
  struct Exception 
  /*: public std::exception */ {
    Exception(char const * imess) : mess(imess){}
    ~Exception() /* throw */ {}
    char const * what() const /* throw */ {
      return mess.c_str();
    }
    std::string mess;
  };

  K(int ik) : k(ik) {}

  void print() {
    auto lk =getK();
    Lock l(global::coutLock);
    std::cout << "K in "
	      << std::this_thread::get_id() 
      //	      << " is " << getK() << std::endl;
	      << " is " << lk << std::endl;
  }

  void setK(int ik) {
    // verify(ik);
    Lock l(lock);
    verify(ik);
    k = ik;
  }

  int getK() const {
    Lock l(lock);
    return k;
  }

 void verify(int ik) const {
   verifyGreater(ik);
 }

  void verifyPositive(int ik) const {
    if (ik>0) return;
    Lock l(global::coutLock);
    std::cout << "in K::verify " 
	      << ik << " not positive " << std::endl;
    throw Exception("input non positive");
  }

  void verifyGreater(int ik) const {
    if (ik>k) return;
    Lock l(global::coutLock);
    std::cout << "in K::verify " 
	      << ik << " not greater than " << k << std::endl;
    throw Exception("input too small");
  }

private:
  mutable Mutex lock;
  int k;

};


void setK( K& k) {
  
  int j=0;
  for (int i=0; i<1000; i++) {
    j = (j>0) ? -(j+1) : -(j-1);
    try {
      k.setK(j);
    } catch( K::Exception const & ce) {
      Lock l(global::coutLock);
      std::cout << ce.what() << std::endl;
    }
  }
  
}

void getK( K& k, int n) {
  
  int j=0;
  for (int i=0; i<n; i++) {
    try {
      k.print();
      // Lock l(global::coutLock);
      // std::cout << i << " " << k.getK() << std::endl;
    } catch( K::Exception const & ce) {
      Lock l(global::coutLock);
      std::cout << ce.what() << std::endl;
    }
  }
  
}


void deadlock() {

  K k(5);

  getK(k,1);

  Thread t1(setK,std::ref(k));
  Thread t2(getK,std::ref(k),5000);
  Thread t3(getK,std::ref(k),5000);

  getK(k,5000);

  t1.join();
  t2.join();
  t3.join();
  
  getK(k,1);

}

int main() {

  hello();

  // deadlock();

  return 0;
}



