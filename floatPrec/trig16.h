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
  HOST_DEVICE_CONSTANT uint16_t mask = 3<<14;
  
  using Sin14 = LUT<14,std::bit_cast<uint32_t>(pi2)>;
  using Sin15 = LUT<15,std::bit_cast<uint32_t>(pi)>;
  HD_INLINE int16_t to16(float x) { return Sin15::toi(x);}
  HD_INLINE float tof(int16_t x) { return  Sin15::tof(x);}

  Sin14  sin14(std::sin<float>);

  using  Lut10 = LUT<10,std::bit_cast<uint32_t>(pi32)>;
  Lut10  sin10(std::sin<float>);
  Lut10  cos10(std::cos<float>);
  constexpr float tick[5] = {0, 2*pi32, 4*pi32, 6*pi32, 8*pi32};


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
