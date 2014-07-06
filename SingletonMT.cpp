#include <memory>

#include<iostream>
#include <thread>
#include <functional>
#include <algorithm>
#include<vector>
#include<cstdatomic>
// convert gcc to c0xx
#define thread_local __thread

typedef std::thread Thread;
typedef std::vector<std::thread> ThreadGroup;
typedef std::mutex Mutex;
typedef std::unique_lock<std::mutex> Guard;
typedef std::condition_variable Condition;

template <typename T>
class Singleton {
private:
  typedef Singleton<T> self;

  mutable std::once_flag m_flag;
  mutable std::unique_ptr<T> m_data;
  std::atomic<bool> ready;

  Singleton() : ready(false){
    init();
    ready=true;
 }
  
  void init() const {
    m_data.reset(new T);
  }

public:
  T const & data() const {
    // std::call_once(m_flag,&self::init,this);
    return *m_data;
  }

public:
  static self & instance() {
    static self me;
    // while (!me.ready) nanosleep(0,0);
    return me;
  }

 
};

struct Int {
  static std::atomic<int> count;
  Int() : me(4){count++;}
  ~Int() { std::cout << "bye" << std::endl;}
  int me;
};
std::atomic<int> Int::count(0);


struct Client {
  static std::atomic<int> start;
  static std::atomic<int> error;
  ~Client() {
    // std::cout << "destr " << d << " " << (&(Singleton<Int>::instance().data().me)) << std::endl;
    if(d!=(&(Singleton<Int>::instance().data().me))) ++error;
  }

  static void wait() {
    --start;
    do{}while(start);
  }
  void operator()() {
    // wait everybody is ready;
    wait();
    d = (&(Singleton<Int>::instance().data().me));
    // std::cout << "op " << d << " " << (&(Singleton<Int>::instance().data().me)) << std::endl;

  }

  int const * d;
};

std::atomic<int> Client::start(0);
std::atomic<int> Client::error(0);


int main() {


  int NUMTHREADS = 8;
  Client::start=NUMTHREADS+1;
  ThreadGroup threads;
  threads.reserve(NUMTHREADS);
  for (int i=0; i<NUMTHREADS; ++i) {
    threads.push_back(Thread(Client()));
  }
   Client::error=0;
  std::cout <<  Client::start << std::endl;
  Client::wait();
  std::for_each(threads.begin(),threads.end(), 
		std::bind(&Thread::join,std::placeholders::_1));




  std::cout << Int::count << std::endl;
  std::cout << Client::error << std::endl;
  return 0;


}
