#include <cstdint>
#include <iostream>


int main() {

  int imax = 256;
  auto mask = [&]()->uint32_t { return imax-1;};
  
  for (uint32_t i=0; i<1024; ++i) {
   std::cout << int(i&mask()) << ' ';
  }

  std::cout << std::endl;
  return 0;
}
