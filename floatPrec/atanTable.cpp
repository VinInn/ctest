#include "trig16.h"
#include<iostream>


HD_INLINE float setMantissa(float x, uint32_t m) {
   return std::bit_cast<float>(std::bit_cast<uint32_t>(x)|m);
} 


int main() {

   std::cout << setMantissa(1.f,1<<4) - 1.f << std::endl;
   std::cout << setMantissa(1.f,1<<5) - 1.f << std::endl;
   std::cout << setMantissa(1.f,1<<6) - 1.f << std::endl;
   std::cout << setMantissa(1.f,1<<7) - 1.f << std::endl;
   std::cout << setMantissa(1.f,1<<8) - 1.f << std::endl;
   std::cout << setMantissa(1.f,32767<<7) - 1.f << std::endl;
   std::cout << setMantissa(1.f,32767<<8) - 1.f << std::endl;

   {
   int k=8192;
   int dup=0; int skip=0;
   for (int i=0; i<32768; ++i) {
     float x = setMantissa(1.f,i<<8) - 1.f;
     float a = -std::atan((x-1)/(x+1));
     uint16_t j = trig16::to16(a);
     assert(j<=8192);
     if (j==k) dup++;
     else if (j!=(k-1)) skip++;
     k=j;
   }
  std::cout << "15: " << dup << ' ' << skip << std::endl;
  }

  {
   int k=8192;
   int dup=0; int skip=0;
   for (int i=0; i<32768/2; ++i) {
     float x = setMantissa(1.f,i<<9) - 1.f;
     float a = -std::atan((x-1)/(x+1));
     uint16_t j = trig16::to16(a);
     assert(j<=8192);
     if (j==k) dup++;
     else if (j!=(k-1)) skip++;
     k=j;
   }
  std::cout << "14: " << dup << ' ' << skip << std::endl;
  }

  {
   int k=8192;
   int dup=0; int skip=0;
   for (int i=0; i<32768/4; ++i) {
     float x = setMantissa(1.f,i<<10) - 1.f;
     float a = -std::atan((x-1)/(x+1));
     uint16_t j = trig16::to16(a);
     assert(j<=8192);
     if (j==k) dup++;
     else if (j!=(k-1)) skip++;
     k=j;
   }
  std::cout << "13: " << dup << ' ' << skip << std::endl;
  }

  {
   int k=8192;
   int dup=0; int skip=0;
   for (int i=0; i<32768/8; ++i) {
     float x = setMantissa(1.f,i<<11) - 1.f;
     float a = -std::atan((x-1)/(x+1));
     uint16_t j = trig16::to16(a);
     assert(j<=8192);
     if (j==k) dup++;
     else if (j!=(k-1)) skip++;
     k=j;
   }
  std::cout << "12: " << dup << ' ' << skip << std::endl;
  }


std::cout << std::endl;

  int nerr=0; int w=0;
  for (int i=-32768; i<32768; ++i) {
    for (float z = -0.7e-5f; z< 0.71e-5f; z+=0.35e-5f) {
      float y = std::sin(trig16::tof(i)+z);
      float x = std::cos(trig16::tof(i)+z);
      auto a = trig16::atan216(y,x);
      if (std::abs(i-a)>0) nerr++;
      if (std::abs(i-a)>1) std::cout << i << ' ' << x << ','<<y << ' ' << z << ' '
         << (std::abs(x) - std::abs(y))/(std::abs(x) + std::abs(y)) << ' ' << a << ' ' << i-a << std::endl;
      w++;
    }
  }
  std::cout << "atan2 err " << nerr << " / " << w << std::endl;

  return 0.;
}

