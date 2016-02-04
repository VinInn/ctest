#include <cstring>

enum class Num { zero, one, two, three };

Num conv1(int const i) {
  Num res; memcpy(&res,&i,sizeof(i)); return res;
}

Num conv2(int const i) {
   return static_cast<Num>(i);
}

Num num;

#include<iostream>
int main() {

  std::cout << static_cast<int>(conv1(0)) << ' ' << static_cast<int>(conv1(5)) << std::endl;
  std::cout << static_cast<int>(conv2(0)) << ' ' << static_cast<int>(conv2(5)) << std::endl;

  switch (num) {
    case Num::three :
       std::cout<< "three" << std::endl;
       break;
    case Num::zero : 
       std::cout<< "zero" << std::endl;
       break;
    case Num::one : 
        std::cout<< "one" << std::endl;
      break;
  }

  return 0;
}
