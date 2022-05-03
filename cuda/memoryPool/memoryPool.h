#pragma once
#include<memory>
#include<new>


class FastPoolAllocator;

namespace memoryPool {

  struct DeleterBase {

    virtual ~DeleterBase() = default;
    virtual void operator()(int bucket) =0;

  };

  class Deleter {
  public:
    explicit Deleter(int bucket=-1) : m_bucket(bucket) {}
    Deleter(std::shared_ptr<DeleterBase> del, int bucket=-1) : me(del), m_bucket(bucket) {}

    void set(std::shared_ptr<DeleterBase> del) { me=del;}

    void operator()(void * p) {
      if (!me) throw std::bad_alloc(); 
      (*me)(m_bucket);
    }

    std::shared_ptr<DeleterBase> me;
    int m_bucket;
  };


  template <typename T>
  using unique_ptr = std::unique_ptr<T,Deleter>;


}
