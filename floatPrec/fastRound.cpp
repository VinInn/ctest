#include<cstdint>

template<typename T>
struct FastRoundTrait{
};


template<>
struct FastRoundTrait<float>{
 static constexpr float magic = 12582912.f; // 2^23 +2^22
static constexpr int shift = 10;
 using INT = int32_t;
 union fi { constexpr fi(float q):f(q){} float f; INT i;};
};


template<>
struct FastRoundTrait<double>{
 static constexpr float magic = 6755399441055744.; // 2^52 +2^51
 static constexpr int shift = 13;
 using INT = int64_t;
 union fi { constexpr fi(double q):f(q){}  double f; INT i;};
};


template<typename T>
constexpr typename FastRoundTrait<T>::INT fastRound(T t) {
  typename FastRoundTrait<T>::fi fi(t);
  fi.f +=FastRoundTrait<T>::magic;
  // no need to shift (take lower N/2 bits) if number is small
  return (fi.i<<FastRoundTrait<T>::shift)>>FastRoundTrait<T>::shift;
}


// Based on code from 
// https://graphics.stanford.edu/~seander/bithacks.html
constexpr int count_leading_zeroes(uint64_t v) 
{ 
  constexpr char bit_position[64] = {  
     0,  1,  2,  7,  3, 13,  8, 19,  4, 25, 14, 28,  9, 34, 20, 40, 
     5, 17, 26, 38, 15, 46, 29, 48, 10, 31, 35, 54, 21, 50, 41, 57, 
    63,  6, 12, 18, 24, 27, 33, 39, 16, 37, 45, 47, 30, 53, 49, 56, 
    62, 11, 23, 32, 36, 44, 52, 55, 61, 22, 43, 51, 60, 42, 59, 58 }; 
    
  v |= v >> 1; // first round down to one less than a power of 2  
  v |= v >> 2; 
  v |= v >> 4; 
  v |= v >> 8; 
  v |= v >> 16; 
  v |= v >> 32; 
  v = (v >> 1) + 1; 
    
  return 63 - bit_position[(v * 0x0218a392cd3d5dbf)>>58]; 
}
  
constexpr uint32_t bits(float f) 
{ 
  if (f == 0.0f) 
    return 0; // also matches -0.0f and gives wrong result 
/*
  else if (f == INFINITY) 
    return 0x7f800000; 
  else if (f == -INFINITY) 
    return 0xff800000; 
  else if (f != f) // NaN 
    return 0x7fc00000; // This is my NaN...
*/  

  bool sign = f < 0.0f; 
  float abs_f = sign ? -f : f; 
  
  int exponent = 254; 
  
  while(abs_f < 0x1p87f) 
  { 
    abs_f *= 0x1p41f; 
    exponent -= 41; 
  } 
  
  uint64_t a = (uint64_t)(abs_f * 0x1p-64f); 
  int lz = count_leading_zeroes(a);
  exponent -= lz;
  
  if (exponent <= 0) 
  { 
    exponent = 0; 
    lz = 8 - 1;
  } 
  
  uint32_t significand = (a << (lz + 1)) >> (64 - 23);
  return (sign << 31) | (exponent << 23) | significand; 
}


constexpr int32_t slowRound(float t) {
  // bool sign = t<0;
  constexpr float magic = 12582912.f; // 2^23 +2^2
  // constexpr uint32_t mask = (~0U<<11)>>11;
  return (int32_t(bits(t+magic))<<10)>>10;
}


#include<iostream>

// #define constexpr const
int main() {


  constexpr int32_t f1 = slowRound(131567.32f);
  constexpr int32_t f2 = slowRound(-131567.32f);
  constexpr int32_t f3 = slowRound(131567.72f);
  constexpr int32_t f4 = slowRound(-131567.72f);

  constexpr auto ff = f3+f4;

  std::cout << f1 << std::endl;
  std::cout << f2 << std::endl;
  std::cout << f3 << std::endl;
  std::cout << f4 << std::endl;
  std::cout << ff << std::endl;

  return 0;

}

 
