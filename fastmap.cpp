/* a trivial hash-map */



#include<vector>


template<typename K, typename V, typename H>
class FastMap {

public:
  typedef K key_type;
  typedef V value_type;
  typedef H hash_fun;

  explicit FastMap(size_t s=1024) : buckets(s){}
  explicit FastMap(value_type const & v, size_t s=1024) : buckets(s,Node(v)){}

  struct Node {
    Node() : hash(0){}
    explicit Node(value_type const & v) : value(v), hash(0){}
    volatile key_type key;
    volatile value_type value;
    volatile size_t hash; // unique hash....
  };
  
  
  Node & get(key_type const & key) {
    size_t hash = hfun(key);
    size_t idx = hash & ( buckets.size()-1);
    bool full=false;
    while (true) {
      volatile size_t & h = buckets[idx].hash;
      if (h==0 &&  __sync_bool_compare_and_swap(&h,0,hash)) {
	buckets[idx].key=key; // does not matter if collision as long as hash is unique
	break;
      }
      if (h==hash) break;
      idx++; // reprobe
      if (idx==buckets.size()) {
	if (full) throw "no more space";
	idx=0;
	full=true;
      }
    }
    return  buckets[idx];
  }

  // need a real iterator that skips empty buckets...
  typedef typename std::vector<Node>::const_iterator const_iterator;

  std::vector<Node> buckets;
  hash_fun hfun;
  
};

#include <iostream>

template<typename K, typename V, typename H>
void dump(FastMap< K, V,H> const & map) {
  typename FastMap< K, V,H>::const_iterator b=map.buckets.begin();
  size_t size=0;
  while (b!=map.buckets.end()) {
    if ((*b).hash!=0) {
      ++size;
      std::cout << size << ": " << (*b).key << ", " << (*b).value << std::endl;
    }
    ++b;
  }

}


struct hashFun {
  size_t operator()(size_t v) const { return v;}
};


struct Filler {
  static volatile long start;
  typedef FastMap<size_t, size_t, hashFun> Map;
  Filler() : map(0,1024){}
  void wait() {
    __sync_add_and_fetch(&start,-1);
    do{}while(start);
  }
  void operator()() {
    // wait everybody is ready;
    wait();
    // strict collision....
    for (int i=1; i<2000; i=i+2) {
      Map::Node & n = map.get(i);
      __sync_add_and_fetch(&n.value,1);
    }
  }
  Map map;
};

volatile long Filler::start=0;

#include <thread>
#include <functional>
#include <algorithm>

// convert gcc to c0xx
#define thread_local __thread;

typedef std::thread Thread;
typedef std::vector<std::thread> ThreadGroup;
typedef std::mutex Mutex;
typedef std::unique_lock<std::mutex> Guard;
typedef std::condition_variable Condition;



int main() {
  
  typedef FastMap<size_t, size_t, hashFun> Map;
  Map map(0, 1024);
  
  {
    Map::Node & n = map.get(47);
    __sync_add_and_fetch(&n.value,1);
  }
  {
    Map::Node & n = map.get(1024+47);
    __sync_add_and_fetch(&n.value,1);
  }
  dump(map);
  
  const int NUMTHREADS=10;
  Filler::start=NUMTHREADS;
  Filler filler;
  
  ThreadGroup threads;
  threads.reserve(NUMTHREADS);
  for (int i=0; i<NUMTHREADS; ++i) {
    threads.push_back(Thread(std::ref(filler)));
  }
  
  std::for_each(threads.begin(),threads.end(), 
		std::bind(&Thread::join,std::placeholders::_1));
  
  dump(filler.map);
  
  return 0;
}



