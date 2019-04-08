#ifndef CircularQueue_H
#define CircularQueue_H

#include <cstdint>
#include <limits>



// a thread safe lock-free circular queue
// will never work, easy to prove
template<typename T, int MAXSIZE> 
class CircularQueue {
public:

  using ull_t = unsigned long long;

  struct HeadTail {
    uint32_t m_head;
    uint32_t m_tail;
  };

  static constexpr ull_t one() { return ull_t(1)<<32;}
  static constexpr uint32_t head(ull_t ht) { return ((HeadTail&)(ht)).m_head;}
  static constexpr uint32_t tail(ull_t ht) { return ((HeadTail&)(ht)).m_tail;}


#ifdef TEST_CIRCULAR_QUEUE
  static constexpr uint32_t maxCapacity = 8*1024;
  static constexpr uint32_t maxHead =     16*1024-1;

#else
  static constexpr uint32_t maxCapacity = std::numeric_limits<uint32_t>::max()/4+1;
  static constexpr uint32_t maxHead = std::numeric_limits<uint32_t>::max()/2;
#endif

  __device__
  bool constructEmpty(uint32_t capacity) {
   m_capacity = capacity;
   tail()=head()=0;
   return m_capacity<=MAXSIZE;
  }

  __device__
  bool constructFull(uint32_t capacity) {
   m_capacity = capacity;
   tail()=0;
   head()=m_capacity;
   return m_capacity<=MAXSIZE;
  }

  // back to minimal range so that tail <= head
  __device__
  void reset() {
    auto old = m_ht;
    auto val = old;
    int s = int(val.m_head)-int(val.m_tail);
    val.m_tail &= mask();
    val.m_head = val.m_tail +s;

    while (head()>maxHead && ((ull_t const&)old)!=atomicCAS(headTail(),(ull_t const&)old,(ull_t const&)val)) {
      old = m_ht;
      auto val = old;
      int s = int(val.m_head)-int(val.m_tail);
      val.m_tail &= mask();
      val.m_head = val.m_tail +s;
    }
  }


  // we assume we push only what already pop, overflow not possible
  __device__
  void unsafePush(const T &element) {
    auto previous = head(atomicAdd(headTail(), 1));
    previous &= mask();
    m_data[previous] = element;
    if (head()>maxHead) reset(); // avoid wrapping
  }

  __device__
  bool push(const T &element) {
    int t = tail();
    auto previous = head(atomicAdd(headTail(), 1));
    if (int(previous)-t < int(capacity())) {
      previous &= mask();
      m_data[previous] = element;
      if (head()>maxHead) reset(); // avoid wrapping
      return true;
    } else {
      atomicAdd(headTail(), -(1UL));
      return false;
    }
  }

  __device__
  T pop(T const invalid) {
    auto h = head();
    auto previous = tail(atomicAdd(headTail(), one()));
    if (previous<h) { 
      previous &= mask();
      auto val = m_data[previous];
//      while(invalid==val) {val = m_data[previous];}
      m_data[previous] = invalid;
      return val;
    } else {
      atomicAdd(headTail(), -one());
      return invalid;
    }
  }

  inline constexpr int size() const { return int(head())-int(tail());}
  inline constexpr bool empty() const { return tail()>=head();}
  inline constexpr bool full()  const { return size()>=int(capacity());}
  inline constexpr uint32_t capacity() const { return m_capacity; }
  inline constexpr T const * data() const { return m_data; }
  inline constexpr T * data()  { return m_data; }

  inline constexpr uint32_t mask() const { return m_capacity-1;}
  inline constexpr ull_t mask2() const { return ull_t(m_capacity-1)<<32 | ull_t(m_capacity-1);}

  inline constexpr uint32_t head() const { return m_ht.m_head;}
  inline constexpr uint32_t tail() const { return m_ht.m_tail;}
  inline constexpr uint32_t & head()  { return m_ht.m_head;}
  inline constexpr uint32_t & tail() { return m_ht.m_tail;}

  inline constexpr ull_t * headTail() { return (ull_t *)(&m_ht);}
 
private:

  HeadTail m_ht;
  uint32_t m_capacity;  // must be power of 2

  T m_data[MAXSIZE];

};


#endif
