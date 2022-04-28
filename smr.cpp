#include<atomic>
#include<memory>
#include<algorithm>
#include<cassert>
#include<mutex>
#include <vector>


// a sequential id limited to the maximum number of concurrent threads....
// actually just an id limited to the maximum number of active users...
class ThreadId {
public:

  int get() {
    std::lock_guard<std::mutex> lock(m_mutex);
    ++m_active;
    if (m_freeList.empty()) {
      return m_total++;    
    } 
    auto ret = m_freeList.back();
    m_freeList.pop_back();
    return ret;    
  }

  void redeem(int id) {
    std::lock_guard<std::mutex> lock(m_mutex);
    assert(m_active>0);
    assert(m_total>0);
    assert(id<m_total);
    --m_active;
    if ((id+1)==m_total) --m_total;
    else m_freeList.push_back(id);
  }


private:
  std::mutex m_mutex;
  std::vector<int> m_freeList;
  int m_total=0;  
  int m_active=0;
};

#include<iostream>

ThreadId threadId;

template<typename T, int NT>
class SMR {


public:
  static constexpr int maxThreads = NT;
  static constexpr int clsize = 8; //  std::hardware_destructive_interference_size/sizeof(T*)
  static constexpr int scanThreshold = 2*maxThreads;
  static constexpr int maxHP = maxThreads;  


  static SMR & instance(T** hp) {
    thread_local std::unique_ptr<SMR> me(new SMR(hp));
    return *me;
  }

  class Holder {
  public:     
    void insert(T * node) {
     m_hp[SMR::instance(m_hp).iThread()] = node;
    }

   void remove(T * node) {
     m_hp[SMR::instance(m_hp).iThread()] = nullptr;
     SMR::instance(m_hp).deleteNode(node);
   }
  private:

    T* m_hp[maxHP];

  };


  explicit SMR(T** hp) : m_hp(hp), m_thread(threadId.get()) { m_dlist.reserve(scanThreshold);  assert(iThread()<maxThreads);}

  ~SMR()  { scan(); threadId.redeem(iThread()); }

  int iThread() const { return m_thread;}

  void deleteNode(T * node) {
   m_dlist.push_back(node);
   if (scanThreshold==m_dlist.size()) scan();
  }

private:

 void scan() {
   T * plist[maxHP];
   int np=0;

   for (auto p : m_hp) if(p) {
     plist[np++]=p;
     std::push_heap(plist,plist+np);
   }
   sort_heap(plist,plist+np);
 
   std::vector<T*> new_dlist;
   new_dlist.reserve(scanThreshold);
   for (auto p : m_dlist) {
     if (std::binary_search(plist,plist+np,p))
       new_dlist.push_back(p);
     else
      delete p;
   }   

   std::swap(m_dlist,new_dlist);

 }


 T** m_hp;
 int m_thread;
 std::vector<T*> m_dlist;
};


template<typename T>
class SafeStack {

  struct Node{
     Node * next;
     T value;
  };

  void push(T const & v) {
    // trivially safe
    auto n = new Node;
    n.value=v;
    n.next = m_top;
    while(!m_top.compare_exchange_weak(n->next, n));
  }

  T pop() {
    // ABA safe
    Node * t = m_top;
    while (true) {
      if (!t) break;
      smr.insert(t);
      if (m_top.compare_exchange_weak(t,t.next)) break;
      t = m_top;
    }
    auto v = t.value();
    smr.remove(t);
    return v;
  }

  bool empty() const { return m_top;}
  T const & top() const { return m_top.value;}
  T & top() { return m_top.value;}


private:

  std::atomic<Node*> m_top = nullptr;

  typename SMR<Node,256>::Holder smr;
};



#include<iostream>

#include<thread>

typedef std::thread Thread;
typedef std::vector<std::thread> ThreadGroup;
typedef std::mutex Mutex;
typedef std::lock_guard<std::mutex> Lock;

int main() {

  using SMRI = SMR<int,256>;

  SMRI::Holder smr;
  auto go = [&] {
     int k = 4;
     smr.insert(&k);
  };

  Thread t0(go);
  const int NUMTHREADS=10;
  ThreadGroup threads;
  threads.reserve(NUMTHREADS);

  for (int i=0; i<3; i++) {
    Thread t(go);
    t.join();

    for (int i=0; i<NUMTHREADS; ++i) {
      threads.emplace_back(go);
    }

    for (auto & t : threads) t.join();
   
    threads.clear();

  }
  
  std::cout << "now the final " << std::endl;

  t0.join();

  Thread t(go);
  t.join();


};
