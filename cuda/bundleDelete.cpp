#include<memory>
#include<array>
#include<atomic>

#include<iostream>

struct BundleDeleteBase {

  virtual ~BundleDeleteBase() = default;
  virtual void operator()(void * p) =0;

};

template<int N>
struct BundleDeleteImpl final : BundleDeleteBase {

  ~BundleDeleteImpl() {
      doFree();
   } 

  void doFree() {
      std::cout << "deleting "<< m_count <<std::endl;
      for (auto p : m_p) ::free(p);
  }

  void operator()(void * p) {
    m_p[m_count++] = p;
    std::cout << m_count << std::endl;
    /*
    if (N==m_count){
      doFree();
    }
    */
  }
  int m_count = 0; // can be atomic ok
  std::array<void *,N> m_p = {nullptr};

};


struct BundleDelete {

  BundleDelete() = default;
  BundleDelete(std::shared_ptr<BundleDeleteBase> del) : me(del){}
 
  void operator()(void * p) {
    if (!me) {
      std::cout << "deleting one "<<std::endl;
      ::free(p);
      return;
    }
    (*me)(p);
  }

  std::shared_ptr<BundleDeleteBase> me;
};




#include<cstdint>
#include<cstdlib>
int main(int argc, char** argv) {

  using Pointer = std::unique_ptr<int,BundleDelete>;

  BundleDelete deleter(std::make_shared<BundleDeleteImpl<4>>());

  auto p0 = Pointer((int*)malloc(sizeof(int)),deleter);
  auto p1 = Pointer((int*)malloc(sizeof(int)),deleter);
  if (argc>1) auto p2 = Pointer((int*)malloc(sizeof(int)),deleter);
  auto p3 = Pointer((int*)malloc(sizeof(int)),deleter);

  auto p = Pointer((int*)malloc(sizeof(int)));

  return 0;
}
