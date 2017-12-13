#include <array>
#include <memory>
#include <atomic>
#include <cassert>

template<typename T>
class concurrent_stack {
public:
  

  struct Node {
    template<typename ...Arguments>
    Node(Arguments&&...args) : t(args...),prev(nullptr){}
    T & operator*() {return t;}
    T t;
    std::atomic<Node *> prev;
  };
  
  using NodePtr = std::unique_ptr<Node>;

  concurrent_stack() : head(nullptr), nel(0){}

  ~concurrent_stack() { while(pop()); assert(0==nel);}
    
  
  NodePtr pop() {
    Node * lhead = head;
    if (!lhead) return NodePtr();
    Node * prev = lhead->prev;
    while (!head.compare_exchange_weak(lhead,prev)) { if (!lhead) return NodePtr(); prev = lhead->prev;}
    if (lhead) {
      lhead->prev=nullptr;
      nel-=1;
      assert(nel>=0);
    }
    return NodePtr(lhead);
  }


  void push(NodePtr && np) {
    auto n = np.release();
    Node * lhead = head;
    n->prev = lhead;
    while (!head.compare_exchange_weak(lhead,n)) n->prev = lhead;
    nel+=1; 
  }

  unsigned int size() const { return nel;}

  template<typename ...Arguments>
  static NodePtr 
    make(Arguments&&...args) { return std::make_unique<Node>(args...);}

  std::atomic<Node *> head;
  std::atomic<int> nel;


  
};


#include<iostream>
#include "tbb/task_group.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/concurrent_queue.h"


std::atomic<unsigned int> ni(0);
struct Stateful {

  Stateful(int i) : id(i){if (i>=0) ni+=1;}
  ~Stateful() { std::cout << "Stateful " << id << '/'<<count << ' ' << ni << std::endl; }


  void operator()() { ++count;}
    
  long long count=0;
  int id=-1;
};

namespace {
  using Stack = concurrent_stack<Stateful>;
  Stack stack;

  using QItem = std::unique_ptr<Stateful>;
  using Queue = tbb::concurrent_queue<QItem>;
  Queue queue;
}



int main() {

  {
  Stack::Node anode(-42);

  std::cout << stack.size() << std::endl;

  stack.push(std::move(stack.make(-99)));

  std::cout << stack.size() << std::endl;

  auto n = stack.pop();

  std::cout << stack.size() << std::endl;

  }
  
  std::cerr << "default num of thread " << tbb::task_scheduler_init::default_num_threads() << std::endl;

  //tbb::task_scheduler_init init;  // Automatic number of threads
   tbb::task_scheduler_init init(tbb::task_scheduler_init::default_num_threads());  // Explicit number of threads

   tbb::task_group g;
 
  auto NTasks = 1000*tbb::task_scheduler_init::default_num_threads();
  // not necessarely a good idea but works...
  for (auto i=0;i<NTasks;++i) {
    auto k=i;
    g.run([&,k]{
	auto a = stack.pop();
	if (!a) a = stack.make(k);
	assert(a.get());
	(**a)();
	stack.push(std::move(a));

        QItem q;
	if (!queue.try_pop(q))
	  q = std::make_unique<Stateful>(-k-1);
	assert(q.get());
	(*q)();
	queue.push(std::move(q));

      }
      );
  }
  g.wait();

  std::cout << stack.size() << std::endl;

  return 0;

}
