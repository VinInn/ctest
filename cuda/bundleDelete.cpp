#include<memory>
#include<array>
#include<atomic>

#include<iostream>



template<int N>
struct BundleDeleteImpl {

  void operator()(void * p) {
    m_p[m_count++] = p;
    std::cout << m_count << std::endl;;
    if (N==m_count){
      std::cout << "deleting "<< std::endl;
      for (auto p : m_p) free(p);
    }
  }
  int m_count = 0; // can be atomic ok
  std::array<void *,N> m_p;

};

template<int N>
struct BundleDelete {

  BundleDelete() = default;
  BundleDelete(std::shared_ptr<BundleDeleteImpl<N>> del) : me(del){}
 
  void operator()(void * p) {
    (*me)(p);
  }

  std::shared_ptr<BundleDeleteImpl<N>> me;
};


#include<cstdint>
#include<cstdlib>
int main() {

  using Pointer = std::unique_ptr<int,BundleDelete<4>>;

  BundleDelete<4> deleter(std::make_shared<BundleDeleteImpl<4>>());

  auto p0 = Pointer((int*)malloc(sizeof(int)),deleter);
  auto p1 = Pointer((int*)malloc(sizeof(int)),deleter);
  auto p2 = Pointer((int*)malloc(sizeof(int)),deleter);
  auto p3 = Pointer((int*)malloc(sizeof(int)),deleter);

  return 0;
}
