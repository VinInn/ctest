#include<iostream>
#include <iomanip>
#include <utility>

typedef std::pair<unsigned long long, unsigned long long> uuid;


int main() {

   uuid u(0x1a7f27745fca4a5e, 0xba36d293a87c06e8);

   std::cout << "1a7f27745fca4a5e-ba36d293a87c06e8" << std::endl;
   std::cout << std::hex << u.first << "-" << u.second << std::endl;



  return 0;
}
