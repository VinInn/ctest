#include<iostream>
#include<limits>
#include<iomanip>

int main() {

  int imax = std::numeric_limits<int>::max();
  unsigned int umax = std::numeric_limits<unsigned int>::max();

  
  int nine = 0.9f*imax;
  int mnine = -nine;

  std::cout << std::hex << nine << ' ' << mnine << std::endl;
  
  
  unsigned int unine =  (0.9f)*umax/2.f;
  unsigned int umnine = (1.1f)*umax/2.f;

  std::cout << std::hex << unine << ' ' << umnine << std::endl;


  
  std::cout << float(nine-mnine)/imax << std::endl;
  std::cout << float(mnine-nine)/imax << std::endl;
  
  std::cout << 2.*float((unine-umnine)^0Xffffffff)/umax << std::endl;
  std::cout << 2.*float((umnine-unine)^0)/umax << std::endl;


  
  return 0;
}
