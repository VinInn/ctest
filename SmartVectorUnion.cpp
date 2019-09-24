#include<vector>
#include<variant>
#include<array>
#include<cstdint>

#include<iostream>
#include<cassert>


// a mimimal smart vector that can be either an array or a vector
template<typename T>
class SmartVector {
public :
  using Vector = std::vector<T>;
  static constexpr uint32_t maxSize = sizeof(Vector)/sizeof(T)-1;
  using Array = std::array<T,sizeof(Vector)/sizeof(T)>;
  union Variant {
    // all nop
    Variant(){}
    ~Variant(){}
    Variant(const Variant &){}
    Variant & operator=(const Variant &){ return *this;}
    Variant(Variant&&){}
    Variant & operator=(Variant &&){ return *this;}

    Array a;
    Vector v;
  };

  SmartVector(){
    m_container.a.back()=0;
  }

  ~SmartVector(){
   if(!m_isArray) 
     m_container.v.~Vector();
  }

  SmartVector(const SmartVector& sm) : m_isArray(sm.m_isArray) {
   if(m_isArray) {
    m_container.a=sm.m_container.a;
   } else {
    m_container.v=sm.m_container.v;
   }
  }
  SmartVector& operator=(const SmartVector& sm) {
    m_isArray = sm.m_isArray;
   if(m_isArray) {
    m_container.a=sm.m_container.a;
   } else {
    m_container.v=sm.m_container.v;
   }
   return *this;
  }

  SmartVector(SmartVector&& sm) : m_isArray(sm.m_isArray) {
   if(m_isArray) {
    m_container.a=std::move(sm.m_container.a);
   } else {
    m_container.v=std::move(sm.m_container.v);
   }
  }

  SmartVector& operator=(SmartVector&& sm) {
    m_isArray = sm.m_isArray;
   if(m_isArray) {
    m_container.a=std::move(sm.m_container.a);
   } else {
    m_container.v=std::move(sm.m_container.v);
   }
   return *this;
  }


  template<typename Iter>
  SmartVector(Iter b, Iter e) {
     initialize(b,e);
  }

  template<typename Iter>
  void initialize(Iter b, Iter e) {
     if (e-b<=maxSize) {
       m_isArray=true;
       auto & a = m_container.a;
       std::copy(b,e,a.begin());
       a.back()=e-b;
     } else {
       m_isArray=false;
       std::fill(m_container.a.begin(), m_container.a.end(),0);
       m_container.v.insert(m_container.v.end(),b,e);
    }
  }

  template<typename Iter>
  void extend(Iter b, Iter e) {
    if(m_isArray) {
      auto & a = m_container.a;
      auto cs = a.back();         
      uint32_t ns = (e-b)+cs;
      if (ns<=maxSize) {
        std::copy(b,e,&a[cs]);
        a.back()=ns;
      } else {
        Vector v; v.reserve(ns);
        v.insert(v.end(),m_container.a.begin(),m_container.a.begin()+cs);
        v.insert(v.end(),b,e);
        std::fill(m_container.a.begin(), m_container.a.end(),0);
        m_container.v = std::move(v);
        m_isArray=false;
      }
    }else {
     m_container.v.insert(m_container.v.end(),b,e);
    }
  }


  T const * begin() const {
    if(m_isArray)
       return m_container.a.data();
    else
       return m_container.v.data();
  }

  T const * end() const {
    if(m_isArray)
      return m_container.a.data() + m_container.a.back();
    else
      return  m_container.v.data() + m_container.v.size();
  }

  T const & operator[](uint32_t i) const {
    return *(begin()+i);
  }

  uint32_t size() const {
    if(m_isArray)
       return m_container.a.back();
    else
       return m_container.v.size();
  }


  bool isArray() const { return m_isArray;}
private:
  Variant m_container;
  bool m_isArray = true;
};

int main() {

 using Vector = std::vector<uint8_t>;
 using Array = std::array<uint8_t,sizeof(Vector)/sizeof(uint8_t)>;
 using Variant = SmartVector<uint8_t>;


 std::cout << sizeof(Vector) <<' '<< sizeof(Array) <<' '<< sizeof(Variant) << std::endl;

{
 Variant v;
 assert(v.isArray());
 assert(0==v.size());
}

 uint8_t data[128];
 for (int i=0; i<128; ++i) data[i]=i;

 uint8_t i=0;
 Variant va(data,data+5);
 assert(5==va.size());
 assert(5==va.end()-va.begin());
 assert(va.isArray());
 i=0; for (auto c : va) assert(c==i++);
{
 Variant vb; vb.initialize(data,data+24);
 assert(24==vb.size());
 assert(24==vb.end()-vb.begin());
 assert(!vb.isArray());
 i=0; for (auto c : vb) assert(c==i++);
}
 Variant vv; vv.extend(data,data+64);
 assert(64==vv.size());
 assert(64==vv.end()-vv.begin());
 assert(!vv.isArray());
 i=0; for (auto c : vv) assert(c==i++);

 va.extend(data+5,data+10);
 assert(10==va.size());
 assert(10==va.end()-va.begin());
 assert(va.isArray());
 i=0; for (auto c : va) assert(c==i++);
 va.extend(data+10,data+64);
 assert(64==va.size());
 assert(64==va.end()-va.begin());
 assert(!va.isArray());
 i=0; for (auto c : va) assert(c==i++);

 vv.extend(data+64,data+72); 
 assert(72==vv.size());
 assert(72==vv.end()-vv.begin());
 assert(!va.isArray());
 i=0; for (auto c : vv) assert(c==i++);

 return 0;

}


