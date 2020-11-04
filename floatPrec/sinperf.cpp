//  c++ -Ofast -march=skylake-avx512 sinperf.cpp -fopt-info-vec -mprefer-vector-width=512 -I../../vdt/include/
#include<cmath>
#include "vdtMath.h"

int main() {

  double res=0;

  double x[1024];
  for (int j=0; j<1024; ++j) x[j] = 0.01*j;

  for (int i=0; i<1024*1024; ++i) {
      auto inc = x[1023];
      for (int j=0; j<1024; ++j) x[j]+=inc;
      for (int j=0; j<1024; ++j) {
         auto y = 1./std::sqrt(std::sqrt(0.0001*x[j]));
         auto y1 = std::sin(y);
         auto y2 = vdt::fast_sin(y);
         res+=std::abs(y2-y1);
      }
  }

  return res > 0.5; 

}
