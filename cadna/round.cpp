#include <iostream>
#include <cstdio>
#include <cmath>
#include <fenv.h>

int main(int argc, char * argv[]) {

  float x2 = argc +1.f;
  float x3 = x2+1.f;  

  {
  auto y = std::sqrt(x2);
  auto y1 = -y;
  auto zp = -(y/std::sqrt(x3));
  auto zm = y1/std::sqrt(x3);

  printf("%a %a, %a %a, %a %a\n",x2,x3,y,y1,zp,zm);

  }

  fesetround(FE_UPWARD);


  {
  auto y = std::sqrt(x2);
  auto y1 = -y;
  auto zp = -(y/std::sqrt(x3));
  auto zm = y1/std::sqrt(x3);

  printf("%a %a, %a %a, %a %a\n",x2,x3,y,y1,zp,zm);

  }

}
