#include <fstream>
#include <iostream>


int main() {
  
  while (1) {
    std::ifstream in("goAvx");
    if (in) std::cout << "goAvx" << std::endl;
    else std::cout << "no goAvx" << std::endl;
    in.close();
  }
  return 0;
}
