/*
#include<tuple>
#include<functional>
#include<type_traits>
#include <cstring>
#include <cassert>
*/
#include<type_traits>
#include <atomic>

template<typename T>
class Shared{
public:
  enum  { valueSize = sizeof(T),valueAlign = alignof(T) };
  using cType = unsigned int;
  using aType = std::atomic<unsigned int>;
  using aligned_t = typename std::aligned_storage<valueSize+sizeof(aType),valueAlign>::type;

  Shared(){}
  explicit Shared(Shared & rh) : m_ptr(rh.m_ptr) { incr();}
  Shared(Shared const & rh) : m_ptr(rh.m_ptr) { incr();}
  Shared & operator=(Shared const & rh) { decr(); m_ptr=rh.m_ptr; incr(); return *this;}
  Shared(Shared && rh) : m_ptr(rh.m_ptr) { rh.m_ptr=nullptr;}
  Shared & operator=(Shared && rh) { decr(); m_ptr=rh.m_ptr; rh.m_ptr=nullptr; return *this; }
  


  template<typename ... Args>
  explicit Shared(Args&&... args) : m_ptr(new aligned_t()) {
    new(m_ptr) T(args...); count().store(1,std::memory_order_release);;
  }

  ~Shared(){ decr();}
  
  operator bool() const { return  m_ptr;}

  T * operator->() { return ptr();}
  T & operator *() { return *ptr();}
  T const * operator->() const { return ptr();}
  T const & operator *() const { return *ptr();}
  
  cType use_count() const { return m_ptr ? count().load(std::memory_order_acquire): 0;}
  
  bool unique() const { return m_ptr ? 1==count().load(std::memory_order_acquire): true;}

  T * ptr() { return (T*)(m_ptr); }
  T const * ptr() const { return (T const *)(m_ptr);}

private:

  void destroy() {
    ptr()->~T();
    delete m_ptr; m_ptr=nullptr;
  }

  void decr() {
    if(!m_ptr) return;
    auto count().fetch_
     --count(); if (0==count()) destroy();
  }
  void incr() {
    if(m_ptr) ++count();
  }

  aType & count() const { char * p = (char *)const_cast< aligned_t *>(m_ptr); return *(aType *)(p+valueSize);}
   
  aligned_t * m_ptr=nullptr;
  
};


#include<vector>
#include<string>
#include<iostream>

int main() {

  using VS = std::vector<std::string>;

  using Sh = Shared<VS>;

  Sh a;
  Sh b(1,"hi");

  std::cout << b.use_count() << " " << (*b)[0] << std::endl;

  Sh c(b);

  std::cout << b.use_count() << " " << (*b)[0] << std::endl;

  Sh d(std::move(c));

  std::cout << b.use_count() << " " << (*b)[0] << std::endl;


  return 0;

}
