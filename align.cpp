#include<atomic>
#include<iostream>
#include<vector>


int main() {

  using A = std::atomic<bool>;
  struct AA {alignas(64) std::atomic<bool> v;};

  using V = std::vector<AA>;  

  A a;
 alignas(64) A aa;
 V v(10);

  std::cout << alignof(a) << ' '  << sizeof(a) << std::endl;
  std::cout << alignof(aa) << ' '  << sizeof(aa) << std::endl;

  std::cout << alignof(v[2]) << ' '  << sizeof(v[2]) << std::endl;

  return 0;
}
