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
  using Variant = std::variant<Vector,Array>;

  SmartVector(){}

  template<typename Iter>
  SmartVector(Iter b, Iter e) {
     initialize(b,e);
  }

  template<typename Iter>
  void initialize(Iter b, Iter e) {
     if (e-b<=maxSize) {
       m_container = Array();
       auto & a = std::get<Array>(m_container);
       std::copy(b,e,a.begin());
       a.back()=e-b;
     } else
       m_container. template emplace<Vector>(b,e);
  }

  template<typename Iter>
  void extend(Iter b, Iter e) {
    if(auto pval = std::get_if<Array>(&m_container)) {
      auto cs = pval->back();         
      uint32_t ns = (e-b)+cs;
      if (ns<=maxSize) {
        std::copy(b,e,&(*pval)[cs]);
        pval->back()=ns;
      } else {
        Vector v; v.reserve(ns);
        v.insert(v.end(),pval->begin(),pval->begin()+cs);
        v.insert(v.end(),b,e);
        m_container = std::move(v);
      }
    }else if(auto pval = std::get_if<Vector>(&m_container)) {
      pval->insert(pval->end(),b,e);
    }
    else {
     initialize(b,e);
    }
  }


  T const * begin() const { 
    if(auto pval = std::get_if<Array>(&m_container))
       return pval->data();
    else
       return std::get<Vector>(m_container).data();
  }

  T const * end() const {
    if(auto pval = std::get_if<Array>(&m_container))
       return pval->data()+pval->back();
    else
       return std::get<Vector>(m_container).data()+std::get<Vector>(m_container).size();
  }

  T const & operator[](uint32_t i) const {
    return *(begin()+i);
  }

  uint32_t size() const {
    if(auto pval = std::get_if<Array>(&m_container))
       return pval->back();
    else
       return std::get<Vector>(m_container).size();
  }


  Variant const & container() const { return m_container;}
private:
  Variant m_container;
};

int main() {

 using Vector = std::vector<uint8_t>;
 using Array = std::array<uint8_t,sizeof(Vector)/sizeof(uint8_t)>;
 using Variant = SmartVector<uint8_t>;


 std::cout << sizeof(Vector) <<' '<< sizeof(Array) <<' '<< sizeof(Variant) << std::endl;

 Variant v;

 uint8_t data[128];
 for (int i=0; i<128; ++i) data[i]=i;

 uint8_t i=0;
 Variant va(data,data+5);
 assert(5==va.size());
 assert(5==va.end()-va.begin());
 assert(std::get_if<Array>(&va.container()));
 i=0; for (auto c : va) assert(c==i++);
 Variant vb; vb.initialize(data,data+24);
 assert(24==vb.size());
 assert(24==vb.end()-vb.begin());
 assert(std::get_if<Vector>(&vb.container()));
 i=0; for (auto c : vb) assert(c==i++);
 Variant vv; vv.extend(data,data+64);
 assert(64==vv.size());
 assert(64==vv.end()-vv.begin());
 assert(std::get_if<Vector>(&vv.container()));
 i=0; for (auto c : vv) assert(c==i++);

 va.extend(data+5,data+10);
 assert(10==va.size());
 assert(10==va.end()-va.begin());
 assert(std::get_if<Array>(&va.container()));
 i=0; for (auto c : va) assert(c==i++);
 va.extend(data+10,data+64);
 assert(64==va.size());
 assert(64==va.end()-va.begin());
 assert(std::get_if<Vector>(&va.container()));
 i=0; for (auto c : va) assert(c==i++);

 vv.extend(data+64,data+72); 
 assert(72==vv.size());
 assert(72==vv.end()-vv.begin());
 assert(std::get_if<Vector>(&vv.container()));
 i=0; for (auto c : vv) assert(c==i++);

 return 0;

}


