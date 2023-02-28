#include<cstdint>
#include<limits>
    union DInt {
      double d;
      uint64_t l;
    };
    union FInt {
      float f;
      uint32_t i;
    };

   inline float tof(uint32_t u) { 
     FInt fi;
     // use 23 lsb as mantissa of a float in binade [1,2[
     fi.i = u & 0x007FFFFF;
     fi.i |= 0x3F800000;
     return fi.f -1.f;
   }


  inline float toExp(uint32_t u) {
     FInt fi;
     fi.i = (126-u)<<23;
     return fi.f;     
  }


  inline float norm(uint32_t u) {
   constexpr float den = 1./(1.+std::numeric_limits<uint32_t>::max());
   //constexpr float den = 2.328306158982939e-10; // 1./(1.+std::numeric_limits<uint32_t>::max());
   return den*float(u);
  }

#include "luxFloat.h"

#include <iostream>
#include <iomanip>
#include <ios>
int main(int argc, char * argv[]) {

 std::cout << std::setprecision(9); // std::hexfloat;


  std::cout << __builtin_clz(0UL) << std::endl;
  std::cout << __builtin_clzll(0ULL) << std::endl;

  std::cout << __builtin_clz(1UL<<22) << std::endl;
  std::cout << __builtin_clzll(1ULL<<52) << std::endl;

 bool ext = argc>1;
 auto stretch = [&](float f) { return  2.f*f -1.f ;}; 
 auto print = [&](auto F,uint32_t u) {
  std::cout << ( ext? stretch(F(u)) : F(u)) << ' ';
 };

 auto printNext = [&](auto F,uint32_t u) {
   float const x = ext? stretch(F(u)) : F(u);
   u++;
   while( x== (ext? stretch(F(u)) : F(u))) u++;
   std::cout << "... " << ( ext? stretch(F(u)) : F(u)) << ' '; 
 };
 auto printPrevious = [&](auto F,uint32_t u) {
   float const x = ext? stretch(F(u)) : F(u);
   u--;
   while(x== (ext? stretch(F(u)) : F(u))) u--;
   std::cout << ( ext? stretch(F(u)) : F(u)) << "... ";
 };

 for (uint32_t u=0; u<6; u++) 
   print(tof,u);
 std::cout << std::endl;

 for (uint32_t u=(std::numeric_limits<uint32_t>::max()&0x007FFFFF)/2-1; u<(std::numeric_limits<uint32_t>::max()&0x007FFFFF)/2+4; u++)
   print(tof,u);
 std::cout << std::endl;

 for (uint32_t u=std::numeric_limits<uint32_t>::max()-4; u<std::numeric_limits<uint32_t>::max(); u++)
   print(tof,u);
 print(tof,std::numeric_limits<uint32_t>::max());
 std::cout << std::endl;


 std::cout << std::endl;

 for (uint32_t u=0; u<6; u++)
   print(norm,u);
 printNext(norm,6);
 std::cout << std::endl;

 printPrevious(norm,std::numeric_limits<uint32_t>::max()/2-1);
 for (uint32_t u=std::numeric_limits<uint32_t>::max()/2-1; u<std::numeric_limits<uint32_t>::max()/2+4; u++)
   print(norm,u);
 printNext(norm,std::numeric_limits<uint32_t>::max()/2+4);
 std::cout << std::endl;

 printPrevious(norm,std::numeric_limits<uint32_t>::max()-4);
 for (uint32_t u=std::numeric_limits<uint32_t>::max()-4; u<std::numeric_limits<uint32_t>::max(); u++)
   print(norm,u);
 print(norm,std::numeric_limits<uint32_t>::max());
 std::cout << std::endl;



  std::cout << std::endl;

  std::cout << toExp(1) << std::endl;

  uint32_t k[2]={0,1};
  int i=0;
  auto gen = [&]() { return k[i++];};
  std::cout << luxFloat(gen) << std::endl;
  k[0] = 1<<23; i=0;
  std::cout << luxFloat(gen) << std::endl;

}
