// #include <cstdint>
#include <iostream>


int main() {

  int imax = 256;
  auto mask = [&]() { return imax-1;}
  
  for (unsigned int i=0; i<1024; ++i) {
   std::cout << i&mask << ' ';
  }

  std::cout << std::endl;
  return 0;
}
