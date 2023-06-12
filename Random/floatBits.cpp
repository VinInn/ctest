#include<cstdint>
#include<limits>
#include<cstring>

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

  inline uint32_t f32_to_bits(float x)   { uint32_t u; memcpy(&u,&x,4); return u; }
  inline float f32_from_bits(uint32_t x) { float u;    memcpy(&u,&x,4); return u; }


  inline float norm(uint32_t u) {
   constexpr float den = 1./(1.+std::numeric_limits<uint32_t>::max());
   //constexpr float den = 2.328306158982939e-10; // 1./(1.+std::numeric_limits<uint32_t>::max());
   return den*float(u);
  }


struct FakeGen {
  uint64_t v;
  auto operator()() { return v;}
};

#include "luxFloat.h"

#include <iostream>
#include <iomanip>
#include <ios>
int main(int argc, char * argv[]) {

 std::cout << std::setprecision(9); // std::hexfloat;

 FakeGen fgen;
 std::cout << "n-bits = 21" << std::endl;
 using G21 =  NBitsGen<21,FakeGen>;
 G21  g21(fgen);
 fgen.v = 1; fgen.v |= 2ULL<<21; fgen.v |= 3ULL<<42;
 std::cout << G21::NBits << ' '
           << G21::Shift << ' '
           << G21::NChunks << ' '
           <<  __builtin_popcount(G21::mask) << ' '
           << sizeof(G21::return_type) << ' '
           << std::endl;
 std::cout << g21() << ' ' << g21() << ' ' << g21() << std::endl;


std::cout << "n-bits = 32" << std::endl;
 using G32 =  NBitsGen<32,FakeGen>;
 G32  g32(fgen);
 fgen.v = 1; fgen.v |= 2ULL<<32; fgen.v |= 3ULL<<42;
 std::cout << G32::NBits << ' '
           << G32::Shift << ' '
           << G32::NChunks << ' '
           <<  __builtin_popcount(G32::mask) << ' '
           << sizeof(G32::return_type) << ' '
           << std::endl;
 std::cout << g32() << ' ' << g32() << ' ' << g32() << std::endl;


  std::cout << "\n clz" << std::endl;
  std::cout << __builtin_clz(0UL) << std::endl;
  std::cout << __builtin_clzll(0ULL) << std::endl;

  std::cout << __builtin_clz(1UL<<22) << std::endl;
  std::cout << __builtin_clzll(1ULL<<52) << std::endl;


  std::cout << "\n u->f" << std::endl;
  std::cout << "2^0-1 " << toExp(0) << std::endl;
  std::cout << "2^-1-1 " << toExp(1) << std::endl;
  std::cout << "2^-23-1 " << toExp(23) << std::endl;
  std::cout << tof(0) << std::endl;
  std::cout << tof((1<<23)-1) << std::endl;
  std::cout << "0.5 " << f32_from_bits(126<<23) << std::endl;
  std::cout << "1-e " << f32_from_bits((126<<23)|((1<<23)-1)) << std::endl;
  std::cout << "-0.25 " << f32_from_bits(126<<23)-0.75f << std::endl;
  std::cout << "0.25 " << f32_from_bits((126<<23)|((1<<23)-1))-0.75f << std::endl;

  std::cout << "ff23 " << fastFloat<23>(1ULL<<(64-22))  << ' ' << npower2(23-23) << std::endl;
  std::cout << "ff24 " << fastFloat<24>(1ULL<<(64-24))  << ' ' << npower2(24-23) << std::endl;
  std::cout << "ff41 " << fastFloat<41>(1ULL<<(64-41))  << ' ' << npower2(41-23) << std::endl;
  std::cout << "ff64 " << fastFloat<64>(1)  << ' ' << npower2(64-23) << std::endl;


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
