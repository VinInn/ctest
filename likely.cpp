#if defined(NO_LIKELY)
#define likely(x) (x)
#define unlikely(x) (x)   
#elif defined(REVERSE_LIKELY)
#define unlikely(x) (__builtin_expect(x, true))
#define likely(x) (__builtin_expect(x, false))
#else
#define likely(x) (__builtin_expect(x, true))
#define unlikely(x) (__builtin_expect(x, false))
#endif
 

#include <iostream>

namespace {
  void test(int n) {

    if (likely(n>1)) std::cout << n << std::endl;
    else
      std::cout << "unlike" << std::endl;
    
    if (unlikely(n>1)) std::cout << n << std::endl;
    else
      std::cout << "like" << std::endl;
  }
}

int main() {

  test(0);
  test(2);

  return 0;
}




