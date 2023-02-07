#include<cmath>
#include <iostream>
#include<cstdio>

int main(int h) {

{
  double x =  0x1.01825ca7da7e5p+0;
  x *= h;
  double y = acosh(x);

  std::cout << x << ' ' << y << std::endl;
  printf("%a %a\n",x,y);  

}

// 0x1.0001ff6afc4bap+0 
{
  double x =  0x1.0001ff6afc4bap+0;
  x *= double(h);
  double y = acosh(x);

  std::cout << x << ' ' << y << std::endl;
  printf("%a %a\n",x,y);

}

  double k=0;
  for (double x=1.1; x<1.9; x+=0.00000001) k +=acosh(x);

  std::cout << k << std::endl;
  return k;
}
