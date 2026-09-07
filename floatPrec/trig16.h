#pragma once

#include "LUT16.h"
#include "cstdint"
#include <cmath>
#include <tuple>
#include <cassert>

#include<iostream>

namespace trig16 {
  
  HOST_DEVICE_CONSTANT float pi = M_PI;
  HOST_DEVICE_CONSTANT float cto16 =  (LUT<15>::NBins)/M_PI;
  HOST_DEVICE_CONSTANT float ctof =  M_PI/(LUT<15>::NBins);
  HOST_DEVICE_CONSTANT double pi4d = 0.25*M_PI;
  HOST_DEVICE_CONSTANT double pi2d = 0.5*M_PI;
  HOST_DEVICE_CONSTANT uint16_t mask = 3<<14;
  

  HD_INLINE int16_t to16(float x) { return cto16 * x;}
  HD_INLINE float tof(int16_t x) { return ctof * x;}

  LUT<14> sin14(std::sin<double>,pi2d);
  
  HD_INLINE std::tuple<float,float> sincos(int16_t x) {

    uint16_t q = x&mask;  // quadrant
    uint16_t ss = q>>15; // final sign sin; sign cos before switch
    uint16_t c = (q>>14)&1;  // cos or sin
    uint16_t sc = ss^c;  // final sign cos; sign sin before switch
    int16_t r = x-q;  // back to first quadrant
    assert(r>=0);
    assert(r<LUT<14>::NBins);
    float a = sin14(r);  // sin
    // float b = r==0 ? 1.f : sin14(LUT<14>::NBins - r);  // cos
    float b = sin14(LUT<14>::NBins - r);  // cos
    // std::cout << a << ' ' << b << std::endl;
    a = (sc==0) ? a : -a;
    b = (ss==0) ? b : -b;
    // now move back
    // std::cout << x << ' ' << q << ' ' << s << ' ' << x << ' ' << r << std::endl;
    return {c ? b : a, c ? a : b}; 
  }


}
