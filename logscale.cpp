#include "icsiLog.h"


#include<cmath>
#include<iostream>
#include<cstdlib>

void look(float x) {
  int e;
  float f = ::frexpf(x,&e);
  std::cout << x << " exp " << e << " res " << f << std::endl;

 union {
   float val;
   int x;
 } tmp;
 tmp.val = x;
 const int log_2 = ((tmp.x >> 23) & 255) - 127;//exponent
 tmp.x &= 0x7FFFFF; //mantissa

 std::cout << "exp " << log_2 << " mant as in " << tmp.x << " mant as float " << tmp.val << std::endl;


}

float unpack44(int e, int m) {
  union {
    float val;
    int x;
  } tmp;
  tmp.x = m << (23-4); // mantissa back in place;
  int log_2 = (e-16)+127;
  log_2 = log_2 << 23;
  tmp.x |= log_2;
  return tmp.val;
}

std::pair<int,int> pack44(float x) {
  union {
    float val;
    int x;
  } tmp;
  tmp.val = x;
  const int log_2 = ((tmp.x >> 23) & 255) - 127;//exponent
  tmp.x &= 0x7FFFFF; //mantissa
  tmp.x = tmp.x >> (23-4); //quantize mantissa

  return std::pair<int, int>(log_2+16,tmp.x);

}

float unpack68(int e, int m) {
  union {
    float val;
    int x;
  } tmp;
  tmp.x = m << (23-8); // mantissa back in place;
  int log_2 = (e-64)+127;
  log_2 = log_2 << 23;
  tmp.x |= log_2;
  return tmp.val;
}

std::pair<int,int> pack68(float x) {
  union {
    float val;
    int x;
  } tmp;
  tmp.val = x;
  const int log_2 = ((tmp.x >> 23) & 255) - 127;//exponent
  tmp.x &= 0x7FFFFF; //mantissa
  tmp.x = tmp.x >> (23-8); //quantize mantissa

  return std::pair<int, int>(log_2+64,tmp.x);

}


int main(int i) {


  float probX_units    = 1.0018f;
  float probY_units    = 1.0461f;
  float probX_1_over_log_units = 1.0f / std::log( probX_units );
  float probY_1_over_log_units = 1.0f / std::log( probY_units );

  std::cout <<  log2f(probY_units) << std::endl;
  std::cout <<  std::pow( probY_units, -100) << std::endl;
  std::cout <<  std::pow(2, -100*log2f( probY_units)) << std::endl;
  std::cout <<  std::exp(-100.f*std::log( probY_units)) << std::endl;
  std::cout <<  ldexpf(1., -100*log2f( probY_units)) << std::endl;
  //log2f(0.03)*probY_1_over_log_units


  if (i<2) return 0;


  {
  for (int raw=0; raw!=256; ++raw) {
    float f = std::pow( probY_units, (float)( -raw) );
    std::pair<int,int> res = pack44(f);
    std::cout << raw << " : " << f << " | " << res.first << ", " << res.second << " | " << unpack44(res.first, res.second) << std::endl;
  }
  std::cout << std::endl;

  float f = std::pow( probY_units, -255);
  look(f);
  }
  std::cout << std::endl;
  std::cout << std::endl;

  {
  for (int eraw=0; eraw<16383; eraw+=50)
    for (int i=-2; i<3; ++i) 
    {
      int raw = eraw+i;
      float f = std::pow( probX_units, (float)( -raw) );
      std::pair<int,int> res = pack68(f);
      std::cout << raw << " : " << f << " | "
	<< res.first << ", " << res.second << " | " << unpack68(res.first, res.second) << std::endl;
    }
  std::cout << std::endl;

  float f = std::pow( probX_units, -16383);
  look(f);
  }

  return 0;

}
