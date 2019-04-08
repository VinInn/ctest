#ifndef CircularQueue_H
#define CircularQueue_H

#include <cstdint>
#include <limits>
#include<algorithm>


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

  __device__ __host__
  bool construct(uint32_t capacity) {
   m_capacity = capacity;
   tail()=0;
   head()=m_capacity;
   return m_capacity<=MAXSIZE;
  }



  __device__
  bool push(const T &element) {
    auto h = head();
    auto hmx = end(h);
    auto const inv = invalid();
    while(h<hmx &&  inv!=atomicCAS(m_data+h,inv,element)) {++h;}
    update(0,++h);
    return h<hmx;
  }

  __device__
  T pop() {
    auto t = tail();
    auto tmx = end(t);
    auto const inv = invalid();
    auto val = m_data[t];
    val=atomicCAS(m_data+t,val,inv);
    while(t<tmx && val==inv) { ++t; val = m_data[t]; val=atomicCAS(m_data+t,val,inv); }
    update(1,t+1);
    if (val!=inv) return val;
    if (swap(t)) return pop();
    return inv;
  }

  __device__
  bool update(int loc, uint32_t ind) {
    auto old = m_ht;
    if (!sameSegment(ind,((uint32_t*)(&old))[loc])) return false;
    auto val = old;    
    ((uint32_t*)(&val))[loc] = std::max(((uint32_t*)(&val))[loc],ind);
    while(sameSegment(ind,((uint32_t*)(&old))[loc]) && ((ull_t const&)old)!=atomicCAS(headTail(),(ull_t const&)old,(ull_t const&)val) ) {
      old = m_ht;
      val = old;
      ((uint32_t*)(&val))[loc] = std::max(((uint32_t*)(&val))[loc],ind);
    }
    return sameSegment(ind,((uint32_t*)(&old))[loc]);
  }

  __device__
  bool swap(uint32_t ind) {
    auto old = m_ht;
    if (!sameSegment(ind,old.m_tail)) return true;
    auto val = old;
    val.m_tail = begin(old.m_head);
    val.m_head = begin(old.m_tail);
    auto v = m_data[val.m_tail];
    while(sameSegment(ind,old.m_tail) && ((ull_t const&)(old)) !=atomicCAS(headTail(),(ull_t const&)old,(ull_t const&)val)) {
      old = m_ht;
      val = old;
      val.m_tail = begin(old.m_head);
      val.m_head = begin(old.m_tail);
      v = m_data[val.m_tail];
    }

    return invalid()!=v && !sameSegment(ind,val.m_tail);
  }



  inline constexpr int size() const { return int(head())-int(begin(head()));}
  inline constexpr bool empty() const { return head()==begin(head());}
  inline constexpr bool full()  const { return head()==end(head());}
  inline constexpr uint32_t capacity() const { return m_capacity; }
  inline constexpr T const * data() const { return m_data; }
  inline constexpr T * data()  { return m_data; }

  inline constexpr T invalid() const { return m_data[m_capacity-1];}
  inline constexpr uint32_t segment(uint32_t k) const { return k&capacity();}
  inline constexpr bool sameSegment(uint32_t j,uint32_t k) const { return segment(j)==segment(k);}
  inline constexpr uint32_t begin(uint32_t k) const { return segment(k);}
  inline constexpr uint32_t end(uint32_t k) const { return segment(k) + (m_capacity-1);}

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

  T m_data[2*MAXSIZE];

};


#endif
