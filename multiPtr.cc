#include<atomic>
#include<vector>

namespace edm {
  struct RefBase {};
  template<typename T> struct Ptr{};
  template<typename T> struct Ref{ Ref(RefBase, int); T const & operator *() const; };
}




template<typename T, typename UINT=unsigned short>
class MultiPtrVector {
  private:
    std::vector<edm::RefBase>  collections;
    std::vector<unsigned char>  collIndex;
    std::vector<UINT> index;
    mutable std::atomic<T const *> values;
   
    void push_back(edm::Ptr<T>);
    

    void fill() const;
public:
    ~MultiPtrVector() { delete [] values.load(std::memory_order_relaxed);}
    int size() const { return index.size();}
    T const * begin() const { if (nullptr==values.load(std::memory_order_acquire)) fill(); return values.load(std::memory_order_acquire);}
    T const * end() const { return begin()+size();}
 
     T const & getOne(int i) const { return *edm::Ref<T>(collections[collIndex[i]],index[i]);}
    
};



struct A { int i; float f; std::vector<float> v; };


using V = MultiPtrVector<A>;

int foo(V const & v) {
  return v.end()-v.begin();
}


int sum(V const & v) {
   int s=0;
   for (auto const & k : v) s+= k.i;
   return s;
}
