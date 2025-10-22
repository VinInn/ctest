#include<iostream>

template<typename T>
void foo(T & n, T&m) {
 n = n++ - m++;
}

#include<atomic>
int main() {
  int a = 3;
//  std::atomic<int> a{3};

//   float a=3;  

  std::cout << a << std::endl;
  std::cout << ++a << std::endl;
  std::cout << a++ << std::endl;
  std::cout << a << std::endl;

  foo(a,a);
   std::cout << a << std::endl;

  return 0;

}
