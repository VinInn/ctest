#include <iostream>


int main() {
  unsigned long long l=3;
  for (  int i=0; i<5; i++, l=i )
    std::cout << i << " " << l << std::endl;


  return 0;

 }
