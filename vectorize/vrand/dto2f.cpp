#include<cstdint>
#include<cassert>
#include<iostream>
#include<cmath>

int main() {

  int evenSize = 2*((15+1)/2);
  assert(0==evenSize%2);
  float    rand[evenSize];
  float    x[evenSize];
  union    DInt {
    double d;
    uint64_t l;
  };
  union FInt {
    float f;
    uint32_t i;
  };
  for (int i=0;    i<evenSize; i+=2) {
     DInt di;
     FInt fi;
//     di2.f[0] = 1.f;
     di.d = std::sqrt(double(i+1)/(evenSize-0.1)) +1.;
     //    get 2 float from one double
     assert(di.d<2. && di.d>1.);
     fi.i = (di.l & 0x007FFFFF) | 0x3F800000; // extract mantissa as an FP number
     x[i] = fi.f-1.f;
     assert(x[i]>=0.f && x[i]<=1.f);
     di.l = (di.l >> 23ul);  // remove those used
     //    repeat
     fi.i = (di.l & 0x007FFFFF) | 0x3F800000; // extract mantissa as an FP number
     x[i+1] = fi.f-1.f;
     assert(x[i+1]>=0.f && x[i+1]<=1.f);
  }
  for (int i=0;    i<evenSize; i++) std::cout << x[i] << ' ' ;
  std::cout << std::endl;


  return 0;
}
