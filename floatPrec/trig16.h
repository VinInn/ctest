#pragma once

#include "LUT16.h"
#include "cstdint"
#include <cmath>
#include <tuple>
#include <cassert>

#include<iostream>

namespace trig16 {
  
  HOST_DEVICE_CONSTANT float pi = M_PI;
  HOST_DEVICE_CONSTANT float pi4 = 0.25*M_PI;
  HOST_DEVICE_CONSTANT float pi2 = 0.5*M_PI;
  HOST_DEVICE_CONSTANT float pi32 = 0.03125*M_PI;
  HOST_DEVICE_CONSTANT float pi64 = 0.015625*M_PI;
  HOST_DEVICE_CONSTANT uint16_t mask = 3<<14;
  
  using Sin14 = LUT<14,std::bit_cast<uint32_t>(pi2)>;
  using Sin15 = LUT<15,std::bit_cast<uint32_t>(pi)>;
  HD_INLINE int16_t to16(float x) { return Sin15::toi(x);}
  HD_INLINE float tof(int16_t x) { return  Sin15::tof(x);}

  Sin14  sin14(std::sin<float>);

  using  Lut9 = LUT<9,std::bit_cast<uint32_t>(pi64)>;
  Lut9  sin9(std::sin<float>);
  Lut9  cos9(std::cos<float>);
  HOST_DEVICE_CONSTANT float tick[9] = {0, pi32, 2*pi32, 3*pi32, 4*pi32, 5*pi32, 6*pi32, 7*pi32, 8*pi32};
  HOST_DEVICE_CONSTANT float sinT[9] = {0x0p+0, 0x1.917a6cp-4, 0x1.8f8b84p-3, 0x1.294062p-2, 0x1.87de2cp-2, 0x1.e2b5d4p-2, 0x1.1c73b4p-1, 0x1.44cf34p-1, 0x1.6a09e6p-1};
  HOST_DEVICE_CONSTANT float cosT[9] = {0x1p+0, 0x1.fd88dap-1, 0x1.f6297cp-1, 0x1.e9f416p-1, 0x1.d906bcp-1, 0x1.c38b2ep-1, 0x1.a9b662p-1, 0x1.8bc806p-1, 0x1.6a09e6p-1};
  HOST_DEVICE_CONSTANT float sinPI64 = 0x1.91f66p-5;
  HOST_DEVICE_CONSTANT float cosPI64 = 0x1.ff621ep-1;

  HD_INLINE std::tuple<float,float> sincos(int16_t x) {
    uint16_t q = x&mask;  // quadrant
    uint16_t ss = q>>15; // final sign sin; sign cos before switch
    uint16_t c = (q>>14)&1;  // cos or sin
    uint16_t sc = ss^c;  // final sign cos; sign sin before switch
    int16_t r = x-q;  // back to first quadrant
    // int16_t r = x&(~mask);  // back to first quadrant
    assert(r>=0);
    assert(r<Sin14::NBins);
    float a = sin14(r);  // sin
    float b = r==0 ? 1.f : sin14(Sin14::NBins - r);  // cos
    // std::cout << a << ' ' << b << std::endl;
    a = (sc==0) ? a : -a;
    b = (ss==0) ? b : -b;
    // now move back
    // std::cout << x << ' ' << q << ' ' << sc << ' ' << x << ' ' << r << std::endl;
    return {c ? b : a, c ? a : b}; 
  }


}

#ifdef GENTABLE
#include<iostream>
#include<string>
int main() {

   std::string trail = "  HOST_DEVICE_CONSTANT float ";

   std::cout << std::hexfloat << trail << "sinT[9] = {" << 0.0f;
   for (int i=1; i<9; ++i) std::cout << std::hexfloat << ", " << std::sin(trig16::tick[i]);
   std::cout <<"};" << std::endl;
   std::cout << trail << "cosT[9] = {" << 1.0f;
   for (int i=1; i<9; ++i) std::cout << std::hexfloat << ", " << std::cos(trig16::tick[i]);
   std::cout <<"};" << std::endl; 
   std::cout << std::hexfloat << trail << "sinPI64 = " << std::sin(trig16::pi64) << ';' << std::endl;
   std::cout << std::hexfloat << trail << "cosPI64 = " << std::cos(trig16::pi64) << ';' << std::endl;
   return 0;
}
#endif
