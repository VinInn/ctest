#include <array>
#include <memory>
#include <atomic>
#include <cassert>


struct NodePtr {
  int aba=0;
  int ptr=-1;
  operator bool() const { return ptr>=0;}
};
inline bool operator==(NodePtr a, NodePtr b) { return a.aba==b.aba && a.ptr==b.ptr; }

template<typename T>
class concurrent_stack {
public:
  using Item=T;
  
  struct Node {
    template<typename ...Arguments>
    Node(Arguments&&...args) : t(args...){}
    T & operator*() {return t;}
    T t;
    NodePtr prev;
  };
  
 
  concurrent_stack() : nel(0), ns(0) {}

  ~concurrent_stack() {    
    assert(ns==nel);
    while(pop().ptr>=0); assert(nel==0);
  }
    
  
  NodePtr pop() {
    NodePtr lhead = head;
    //if (lhead.ptr<0) return NodePtr();
    //NodePtr prev = node(lhead)->prev;
    //while (!head.compare_exchange_weak(lhead,prev)) { if(lhead.ptr<0) return NodePtr(); prev = node(lhead)->prev;}
    while (lhead && !head.compare_exchange_weak(lhead,nodes[lhead.ptr]->prev));
    if (lhead.ptr>=0) {
#ifdef VICONC_DEBUG      
      assert(node(lhead)->prev.ptr!=lhead.ptr);
      // assert(node(lhead)->prev==prev);  // usually fails in case of aba
      node(lhead)->prev=NodePtr(); // not needed
#endif
      nel-=1;
#ifdef VICONC_DEBUG      
      // assert(nel>=0);  // may fail 
      verify[lhead.ptr].v-=1;
      assert(0==verify[lhead.ptr].v);
#endif
    }
    return lhead;
  }

  Node * node(NodePtr p) const { return p.ptr>=0 ? nodes[p.ptr].get() : nullptr;}

  void push(NodePtr np) {
#ifdef VICONC_DEBUG      
    verify[np.ptr].v+=1;
    assert(1==verify[np.ptr].v);
#endif
    auto n = node(np);
    np.aba +=1;  // remove this to see aba in action!
    n->prev = head;
    while (!head.compare_exchange_weak(n->prev,np));
    nel+=1;
  }

  unsigned int size() const { return nel;}
  unsigned int nalloc() const { return ns;}

  
  template<typename ...Arguments>
  NodePtr make(Arguments&&...args) {
    int e = ns;
    while (!ns.compare_exchange_weak(e,e+1));
    nodes[e] = std::make_unique<Node>(args...);
#ifdef VICONC_DEBUG      
    verify[e].v=0;
#endif
    return NodePtr{0,e};
  }

  std::atomic<NodePtr> head;
  std::atomic<int> nel;

  std::atomic<int> ns;
  std::array<std::unique_ptr<Node>,1024> nodes;

#ifdef VICONC_DEBUG
  struct alignas(64) IntA64 { int v;};
  std::array<IntA64,1024> verify;
#endif
};

template<typename T>
struct NodePtrGard {
  using Stack = T;
  using Item = typename Stack::Item;
  
  NodePtrGard(Stack & s) : stack(s){
    while(ptr=stack.pop());
  }

  template<typename ...Arguments>
  NodePtrGard(Stack & s,
	      Arguments&&...args
	      ) : stack(s){
    auto a = stack.pop();
    if (!a) a = stack.make(args...);
    assert(a);
    ptr=a;
  }

  ~NodePtrGard() { stack.push(ptr); }

  Item & operator*() {return **stack.node(ptr);}

    
  Stack & stack;
  NodePtr ptr;
};



#include<iostream>
#include "tbb/task_group.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/concurrent_queue.h"


std::atomic<unsigned int> ni(0);
struct Stateful {

  Stateful(int i) : id(i) {if (i>=0) ni+=1;}
  ~Stateful() { std::cout << "Stateful " << id << '/'<<count << ' ' << ni << std::endl; }


  void operator()() {
    ++count;
    
    assert(0==verify);
    ++verify;
    assert(1==verify);
    verify *=2;
    assert(2==verify);
    --verify;
    assert(1==verify);
    --verify;
    assert(0==verify);

  }
    
  long long count=0;
  int id=-1;

  int verify=0;
};

namespace {
  using Stack = concurrent_stack<Stateful>;
  Stack stack;

  using QItem = std::unique_ptr<Stateful>;
  using Queue = tbb::concurrent_queue<QItem>;
  Queue queue;
}



int main() {
#ifdef VICONC_DEBUG
  std::cout << "debug mode" << std::endl;
#endif
  
  {
  Stack::Node anode(-4242);

  std::cout << stack.size() << ' ' << stack.nalloc() << std::endl;

  stack.push(stack.make(4242));

  std::cout << stack.size() << ' ' << stack.nalloc() << std::endl;

  auto n = stack.pop();

  std::cout << stack.size() << ' ' << stack.nalloc() << std::endl;

  stack.push(n);

  std::cout << stack.size() << ' ' << stack.nalloc() << std::endl;

  
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
	{
	  NodePtrGard<Stack> n(stack,k);
	  (*n)();
	}
	  /*
	    auto a = stack.pop();
	    if (!a) a = stack.make(k);
	    auto n = stack.node(a);
	    assert(n);
	    (**n)();
	    stack.push(a);
	  */
	
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

  std::cout << stack.size() << ' ' << stack.nalloc() << std::endl;

  std::cout << queue.unsafe_size() << std::endl;
  
  return 0;

}
