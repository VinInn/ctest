#ifndef CircularQueue_H
#define CircularQueue_H

#include <cstdint>
#include <limits>



// a thread safe lock-free circular queue
template<typename T, int MAXSIZE> 
class CircularQueue {
public:

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
   m_tail=m_head=0;
   return m_capacity<=MAXSIZE;
  }

  __device__
  bool constructFull(uint32_t capacity) {
   m_capacity = capacity;
   m_tail=0;
   m_head=m_capacity;
   return m_capacity<=MAXSIZE;
  }

  // back to minimal range so that m_tail <=m_head
  __device__
  void reset() {
    auto old = m_head;
    while (old==atomicCAS(&m_head,old,m_head&mask2())) {old = m_head;}
    // now pop may fail
    old = m_tail;
    while (old==atomicCAS(&m_tail,old,m_tail&mask2())) {old = m_tail;}
  }


  // we assume we push only what already pop, overflow not possible
  __device__
  void unsafePush(const T &element) {
    auto previous = atomicAdd(&m_head, 1);
    previous &= mask();
    m_data[previous] = element;
    if (m_head>maxHead) reset(); // avoid wrapping
  }

  __device__
  bool push(const T &element) {
    auto previous = atomicAdd(&m_head, 1);
    if (previous-m_tail<capacity()) {
      previous &= mask();
      m_data[previous] = element;
      if (m_head>maxHead) reset(); // avoid wrapping
      return true;
    } else {
      atomicSub(&m_head, 1);
      return false;
    }
  }

  __device__
  T pop(T invalid) {
    auto previous = atomicAdd(&m_tail, 1);
    if (previous<m_head) { 
      previous &= mask();
      return m_data[previous];
    } else {
      atomicSub(&m_tail, 1);
      return invalid;
    }
  }

  inline constexpr int size() const { return m_head-m_tail;}
  inline constexpr bool empty() const { return m_tail>=m_head;}
  inline constexpr bool full()  const { return size()>=capacity();}
  inline constexpr uint32_t capacity() const { return m_capacity; }
  inline constexpr T const * data() const { return m_data; }
  inline constexpr T * data()  { return m_data; }
  inline constexpr uint32_t mask() const { return m_capacity-1;}
  inline constexpr uint32_t mask2() const { return 2*m_capacity-1;}

  inline constexpr uint32_t head() const { return m_head;}
  inline constexpr uint32_t tail() const { return m_tail;}
private:

  uint32_t m_head;
  uint32_t m_tail;
  uint32_t m_capacity;  // must be power of 2

  T m_data[MAXSIZE];

};


#endif
