#include<vector>
#include<variant>
#include<array>
#include<cstdint>

#include<iostream>
#include<cassert>

template<typename T>
class SmartVector {
public :
  using Vector = std::vector<uint8_t>;
  static constexpr uint32_t maxSize = sizeof(Vector)-1;
  using Array = std::array<uint8_t,sizeof(Vector)/sizeof(uint8_t)>;
  using Variant = std::variant<Vector,Array>;

  SmartVector(){}

  template<typename Iter>
  SmartVector(Iter b, Iter e) {
     if (e-b<=maxSize) {
       m_container = Array();
       auto & a = std::get<Array>(m_container);
       std::copy(b,e,a.begin());
       a.back()=e-b;
     } else 
       m_container.emplace<Vector>(b,e);
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
    }else {
      auto & v = std::get<Vector>(m_container);
      v.insert(v.end(),b,e);
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


  uint32_t size() const {
    if(auto pval = std::get_if<Array>(&m_container))
       return pval->back();
    else
       return std::get<Vector>(m_container).size();
  }

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
 std::cout << "va " << va.size() << ' ' << va.end()-va.begin() << std::endl;
 assert(std::get_if<Array>(&va.m_container));
 i=0; for (auto c : va) assert(c==i++);
 Variant vb(data,data+24);
 std::cout << "vb " << vb.size() << ' ' << vb.end()-vb.begin() << std::endl;
 assert(std::get_if<Vector>(&vb.m_container));
 i=0; for (auto c : vb) assert(c==i++);
 Variant vv(data,data+64);
 std::cout << "vv " << vv.size() << ' ' << vv.end()-vv.begin() << std::endl;
 assert(std::get_if<Vector>(&vv.m_container));
 i=0; for (auto c : vv) assert(c==i++);

 va.extend(data+5,data+10);
 std::cout << "va " << va.size() << ' ' << va.end()-va.begin() << std::endl;
 assert(std::get_if<Array>(&va.m_container));
 i=0; for (auto c : va) assert(c==i++);
 va.extend(data+10,data+64);
 std::cout << "va " << va.size() << ' ' << va.end()-va.begin() << std::endl;
 assert(std::get_if<Vector>(&va.m_container));
 i=0; for (auto c : va) assert(c==i++);

 vv.extend(data+64,data+72); 
 std::cout << "vv " << vv.size() << ' ' << vv.end()-vv.begin() << std::endl;
 assert(std::get_if<Vector>(&vv.m_container));
 i=0; for (auto c : vv) assert(c==i++);


}


