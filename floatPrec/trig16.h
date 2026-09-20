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
  HOST_DEVICE_CONSTANT float pi128 = 0.5*pi64;
  HOST_DEVICE_CONSTANT uint16_t mask = 3<<14;

  using Sin14 = LUT<14,std::bit_cast<uint32_t>(pi2)>;
  using Sin15 = LUT<15,std::bit_cast<uint32_t>(pi)>;
  HD_INLINE int16_t to16(float x) { return Sin15::toi(x);}
  HD_INLINE float tof(int16_t x) { return  Sin15::tof(x);}
  using  Lut9 = LUT<9,std::bit_cast<uint32_t>(pi64)>;

  struct Sin {
    HD_INLINE static float operator()(float x) { return sinf(x);}
  };
  struct Cos {
    HD_INLINE static float operator()(float x) { return cosf(x);}
  };

//#ifndef __CUDA_ARCH__
  Sin14  sin14L(Sin{});
  Lut9  sin9L(Sin{});
  Lut9  cos9L(Cos{});
//#endif
#ifdef __NVCC__
  __device__ const Sin14  sin14;
  __device__ const Lut9  sin9;
  __device__ const Lut9  cos9;
#else
 Sin14  sin14(Sin{});
  Lut9  sin9(Sin{});
  Lut9  cos9(Cos{});
#endif

  HOST_DEVICE_CONSTANT float sinT[17] = {0x0p+0, 0x1.91f66p-5, 0x1.917a6cp-4, 0x1.2c8106p-3, 0x1.8f8b84p-3, 0x1.f19f9ap-3, 0x1.294062p-2, 0x1.58f9a8p-2, 0x1.87de2cp-2, 0x1.b5d1p-2, 0x1.e2b5d4p-2, 0x1.07387ap-1, 0x1.1c73b4p-1, 0x1.30ff8p-1, 0x1.44cf34p-1, 0x1.57d694p-1, 0x1.6a09e6p-1};
  HOST_DEVICE_CONSTANT float cosT[17] = {0x1p+0, 0x1.ff621ep-1, 0x1.fd88dap-1, 0x1.fa7558p-1, 0x1.f6297cp-1, 0x1.f0a7fp-1, 0x1.e9f416p-1, 0x1.e2121p-1, 0x1.d906bcp-1, 0x1.ced7bp-1, 0x1.c38b2ep-1, 0x1.b72834p-1, 0x1.a9b662p-1, 0x1.9b3e04p-1, 0x1.8bc806p-1, 0x1.7b5df2p-1, 0x1.6a09e6p-1};
  HOST_DEVICE_CONSTANT float sinC[17] = {0x1.92156p-6, 0x1.2d520ap-4, 0x1.f564e6p-4, 0x1.5e2144p-3, 0x1.c0b826p-3, 0x1.111d28p-2, 0x1.4135cap-2, 0x1.708854p-2, 0x1.9ef796p-2, 0x1.cc66eap-2, 0x1.f8ba5p-2, 0x1.11eb36p-1, 0x1.26d056p-1, 0x1.3affa4p-1, 0x1.4e6caep-1, 0x1.610b76p-1, 0x1.72d084p-1};
  HOST_DEVICE_CONSTANT float cosC[17] = {0x1.ffd886p-1, 0x1.fe9cdap-1, 0x1.fc2648p-1, 0x1.f8765p-1, 0x1.f38f3ap-1, 0x1.ed740ep-1, 0x1.e6288ep-1, 0x1.ddb13cp-1, 0x1.d4134cp-1, 0x1.c954b2p-1, 0x1.bd7c0ap-1, 0x1.b090a4p-1, 0x1.a29a7ap-1, 0x1.93a224p-1, 0x1.83b0ep-1, 0x1.72d084p-1, 0x1.610b74p-1};

  // https://godbolt.org/z/1Yd748rYq
  HD_INLINE float negIf(float x, uint16_t s) { 
    int a=s; a<<=31;
    return  std::bit_cast<float>(a^std::bit_cast<int>(x));
  }


  HD_INLINE std::tuple<float,float> sincos14(int16_t x) {
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
    a = negIf(a,sc);
    b = negIf(b,ss);
    // now move back
    // std::cout << x << ' ' << q << ' ' << sc << ' ' << x << ' ' << r << std::endl;
    return {c ? b : a, c ? a : b}; 
  }


template<bool L=true>
  HD_INLINE std::tuple<float,float> sincos9(int16_t x) {
     uint16_t q = (x&mask);  // quadrant
     int16_t y = x + 16384; // rotate by pi/2
     int16_t z = x - q;  // move to first quadrant
     assert(z>=0);
     assert(z<16384);
     uint16_t  sw = ((x-8192)>>14)&1;
     z  = (z > 8192) ? 16384 -z : z;  // fold around pi/4

     // valid in first octant...
     constexpr uint16_t mask9 = 511;
     constexpr uint16_t mask13 = 15<<9;
     uint16_t r = z&mask9;
     assert(r<512);
     uint16_t bin = (z == 8192) ? 16 : (z&mask13)>>9;
     assert(bin<17);
     float s,c;
     if constexpr (L) {
       s = trig16::sinT[bin]*trig16::cos9(r) + trig16::cosT[bin]*trig16::sin9(r);
       c = trig16::cosT[bin]*trig16::cos9(r) - trig16::sinT[bin]*trig16::sin9(r);
     } else {
       auto sr = tof(r-256);
       auto cr = 1.0f - 0.5f*sr*sr;
       s = trig16::sinC[bin]*cr + trig16::cosC[bin]*sr;
       c = trig16::cosC[bin]*cr - trig16::sinC[bin]*sr;
     } 

     // back to full range
     auto s1 = s;
     s = sw ? s : c;
     c = sw ? c : s1;
     s =  negIf(s,x>>15);  // sin sign
     c =  negIf(c,y>>15);  // cos sign
     return {s,c};
  }


  HD_INLINE float setMantissa(float x, uint32_t m) {
   return std::bit_cast<float>(std::bit_cast<uint32_t>(x)|m);
  }


  constexpr int aShift = 9; //  +1;
  constexpr int aBins= 16384; // /2;
  constexpr float aFromI = 1./aBins;
  constexpr float aToF = aBins;
  HD_INLINE uint16_t atanR(int i) {
    float x = aFromI*float(i);
//    float x = setMantissa(1.f,i<<aShift) - 1.f;
    float a = -std::atan((x-1.f)/(x+1.f));
    int16_t j = trig16::to16(a);
    assert(j>=0);
    assert(j<=8192);
    return j;
  }


struct  Atan13L {
  uint16_t v[aBins+1];
};

Atan13L atan13L;

#ifdef __NVCC__
  __device__ const Atan13L atan13{};
#endif

  struct Gatan {
    Gatan() {
      for (int i=0; i<aBins; ++i) atan13L.v[i] = atanR(i);
      atan13L.v[aBins]=0;
#ifdef __NVCC__
      cudaMemcpyToSymbol(atan13,&atan13L,sizeof(Atan13L));
      cudaMemcpyToSymbol(sin14,&sin14L,sizeof(Sin14));
      cudaMemcpyToSymbol(sin9,&sin9L,sizeof(Lut9));
      cudaMemcpyToSymbol(cos9,&cos9L,sizeof(Lut9));
#endif
    }  
  };

  Gatan gatan;

  HD_INLINE int16_t atan216(float y, float x) {
#ifdef __CUDA_ARCH__
    uint16_t const * atanP = atan13.v;
#else
   uint16_t const * atanP = atan13L.v;
#endif
    auto r = (std::abs(x) - std::abs(y))/(std::abs(x) + std::abs(y));
    /*
    auto q = std::abs(r)+1.f;
    assert(q<=2.f);    
    uint32_t mask13 = (2*aBins-1)<<(aShift-1);
    // round to nearest....
    int32_t a = ((std::bit_cast<uint32_t>(q)&mask13)+(1<<(aShift-1)))>>aShift;
    */
    int32_t a = std::round(std::abs(r)*aToF);
    assert(a>=0);
    assert(a<=aBins);
    // if (q>=2.f) a=aBins;
    auto b = atanP[a] -8192;
    // std::cout << r << ' ' << a << ' ' << b << std::endl;
    if (x<0.0f) r = -r;    
    if (r<0.0f) b = -b;
    auto angle = (x>=0.0f) ? 8192 : 24576;
    angle += b;
    return ( (y < 0.0f)) ? - angle : angle ;
    
    return angle;
  }


}

#ifdef GENTABLE
#include<iostream>
#include<string>
#include<cmath>
int main() {

   std::string trail = "  HOST_DEVICE_CONSTANT float ";
   float pi64 = M_PI/64.;
   float pi128 = M_PI/128.;
   std::cout << std::hexfloat << trail << "sinT[17] = {" << 0.0f;
   for (int i=1; i<17; ++i) std::cout << std::hexfloat << ", " << std::sin(float(i)*pi64);
   std::cout <<"};" << std::endl;
   std::cout << trail << "cosT[17] = {" << 1.0f;
   for (int i=1; i<17; ++i) std::cout << std::hexfloat << ", " << std::cos(float(i)*pi64);
   std::cout <<"};" << std::endl; 
   std::cout << std::hexfloat << trail << "sinC[17] = {" << std::sin(pi128);
   for (int i=1; i<17; ++i) std::cout << std::hexfloat << ", " << std::sin(pi128+float(i)*pi64);
   std::cout <<"};" << std::endl;
   std::cout << trail << "cosC[17] = {" << std::cos(pi128);
   for (int i=1; i<17; ++i) std::cout << std::hexfloat << ", " << std::cos(pi128+float(i)*pi64);
   std::cout <<"};" << std::endl;
   return 0;
}
#endif
