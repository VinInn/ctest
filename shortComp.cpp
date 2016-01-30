#include<iostream>
int main() {
  short a=7,  b=4;

  short x = 5, y = 2;

auto foo = [&]() {
  short c = a>b; c= -c;
  std::cout << c << std::endl;
  short i = (c&x) | ((~c)&y);
  std::cout << i << std::endl;
};

  foo();
  a=-7;
  foo();

  return 0;  

}
