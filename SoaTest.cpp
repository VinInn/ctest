#include<cstdint>
#include<memory>

namespace ntd {
  class seqIter {
  public:
    seqIter(){}
    seqIter(size_t j) : i(j){}
    bool operator!=( seqIter const & rh) const { return i!=rh.i;}
    seqIter & operator++() { ++i; return *this;}
    size_t operator*() const { return i;}
  private:
    size_t i=0;
  };

  class sequence{
  public:
    sequence(size_t is) : m_size(is){}
    seqIter begin() const { return seqIter(0);}
    seqIter end()   const { return seqIter(size());}
    size_t size() const { return m_size;}
  private:
    size_t m_size;
  };

}

//32 bit version....
class Soa {
public:

  typedef uint32_t StorageType;
  // computes the size of the required allocation assuming DOA inherit form Soa and does not have anything else than "Elements"
  template<typename DSOA>
  static constexpr size_t storageSize(size_t is) {
    return is*(sizeof(DSOA)-sizeof(Soa))/sizeof(uint32_t*);
  }

  template<typename T, std::size_t N> class Elem {
    T* offset;
  public:
    Elem(Soa &soa) : offset( (T*)(soa.store(N))){}
    T const& operator[](std::size_t i) const { return offset[i];}
    T & operator[](std::size_t i) { return offset[i];}
  };
  
  Soa(){}
  // externally allocated
  Soa(std::size_t nElems, std::size_t finalSize, uint32_t * storage) : 
    mem(storage,[](uint32_t[]){;}), m_capacity(finalSize),m_size(finalSize){}
  // self allocating
  Soa(std::size_t nElems, std::size_t finalSize) : 
    mem(new uint32_t[finalSize],[](uint32_t p[]){delete[]p;}), m_capacity(finalSize),m_size(finalSize){}
  
  
  ntd::seqIter begin() const { return ntd::seqIter(0);}
  ntd::seqIter end()   const { return ntd::seqIter(size());}


  uint32_t * store(size_t n) { return &mem[n*capacity()];}
  uint32_t capacity() const { return m_capacity;}
  uint32_t size() const { return m_size;}
private:
  std::unique_ptr<uint32_t[],std::function<void (uint32_t[])>> mem;
  uint32_t m_capacity=0;
  uint32_t m_size=0;
  
};


struct MySoa : public Soa {
  
  template<typename... Args>
  explicit MySoa(Args&& ... args) : 
    Soa(2,std::forward<Args>(args)...), eta(*this),phi(*this){}

  Soa::Elem<float,0> eta;
  Soa::Elem<float,1> phi;


};

#include<iostream>

void go(MySoa & soa){
  std::cout << "s " << soa.size() << std::endl;
  for(auto i : soa)  
    soa.eta[i]=i;

  
  for(auto i : soa)  
    std::cout <<" " << i << ": " << soa.eta[i];
  std::cout << std::endl;

  for(auto i : {1,3,7})  
    std::cout <<" " << i << ": " << soa.eta[i];
  std::cout << std::endl;


  auto i=soa.begin();
  auto j=soa.end();
  std::cout <<" " << *j << ": " ;
  if (i!=j) std::cout <<"ok" << std::endl;
  std::cout <<" " << *i << ": " << soa.eta[*i];
  ++i;
  std::cout <<" " << *i << ": " << soa.eta[*i];
  std::cout << std::endl;
  std::cout << std::endl;
}

int main() {

  std::cout << "sizez " << sizeof(MySoa) << " " << sizeof(Soa) << std::endl;
  std::cout << "sizez " << (sizeof(MySoa)-sizeof(Soa))/sizeof(uint32_t*) << std::endl;
  std::cout << "sizez " << Soa::storageSize<MySoa>(1) << std::endl;

  MySoa soa(10);
  go (soa);

  Soa::StorageType lstore[Soa::storageSize<MySoa>(10)]; 
  MySoa soa2(10,lstore);
  go(soa2);

  return 0;

}
