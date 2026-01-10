#include <cstdint>
#include <bit>
struct IEEE32 {
  uint32_t m:23;
  uint32_t e:8;
  uint32_t s:1;
};


IEEE32 toi32(float x) {  return std::bit_cast<IEEE32>(x);}
float toFloat(IEEE32 x) {  return std::bit_cast<float>(x);}


#include<iostream>
int main() {

 auto i = toi32(42.f);
 std::cout << i.s << ' ' << i.e-127 << ' ' << i.m << std::endl;
 i.s = 1;
 std::cout << toFloat(i) << std::endl;

 return 0;
}
