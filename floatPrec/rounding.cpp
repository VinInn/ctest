#include <fenv.h>
#include<iostream>
#include<iomanip>
#include<cmath>

#pragma STDC FENV_ACCESS ON



int main() {

  // store the original rounding mode
  auto originalRounding = fegetround( );

  {
    float pi= std::acos(-1.f);
    std::cout << std::hexfloat << pi << std::endl;
  }
  // establish the desired rounding mode
  fesetround(FE_TOWARDZERO);
  // do whatever you need to do ...
  {
    float pi= std::acos(-1.f); float mpi= -std::acos(-1.f);
    std::cout << std::hexfloat << pi << ' ' << mpi << std::endl;
  }

  fesetround(FE_UPWARD);
  {
    float pi= std::acos(-1.f);
    std::cout << std::hexfloat << pi << std::endl;
  }

  fesetround(FE_DOWNWARD);         
  {
    float pi= std::acos(-1.f);
    std::cout << std::hexfloat << pi << std::endl;
  }


  // ... and restore the original mode afterwards
  fesetround(originalRounding);

   return 0;
}
