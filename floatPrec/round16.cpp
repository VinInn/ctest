#include<iostream>
#include<iomanip>
#include<cmath>
#include<limits>
#include<cstdio>
#include<cstring>

template<typename T>
void print(T x) {
 std::cout<< std::hexfloat << x <<' '<<  std::scientific << std::setprecision(8) << x << ' ' <<  std::defaultfloat << x << std::endl;
}



template<int N> 
float roundit(float x) {
  static_assert(N<23);
  constexpr auto shift = 23-N; 
  constexpr uint32_t mask = 1<<(shift-1);

  uint32_t i; memcpy(&i, &x, sizeof(x));

  i += (i>=0) ? mask : -mask;
  i>>=shift; i<<=shift;
  memcpy(&x, &i, sizeof(x));

  return x;
}



int main() {


  print(roundit<8>(M_PI));
  print(roundit<12>(M_PI));
  print(roundit<16>(M_PI));
  print(roundit<20>(M_PI));
  print(roundit<22>(M_PI));

  print(roundit<8>(-M_PI));
  print(roundit<12>(-M_PI));
  print(roundit<16>(-M_PI));
  print(roundit<20>(-M_PI));
  print(roundit<22>(-M_PI));


  return 0;
}
