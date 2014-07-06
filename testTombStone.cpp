// only 64 bit!
const long tombMask= 0x80000000;
const long pointerMask= 0x7fffffff;

inline bool tombStone(void * volatile p) {
  return ((long)(p)&tombMask)!=0;
}

template<typename T>
inline void setTombStone(T * volatile & p) {
  __sync_or_and_fetch(&p,tombMask);
}

inline void * volatile pointer(void * volatile p) {
  return (void*)((long)(p)&pointerMask);
}


struct A {
  A(): a({1.,2.,3.,4.}){}

  double a[4];
};


#include <iostream>
int main() {

  A * volatile a = new A;
  A * volatile b = a;
  std::cout << a << std::endl;
  std::cout << b << std::endl;

  setTombStone(b);

  std::cout << a << std::endl;
  std::cout << b << std::endl;

  std::cout << pointer(a) << std::endl;
  std::cout << pointer(b) << std::endl;

  std::cout << tombStone(a) << std::endl;
  std::cout << tombStone(b) << std::endl;

  return 0;

};


