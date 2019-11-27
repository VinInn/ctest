#include <iostream>


template<typename T>
void hi() { 
  std::cout <<  __PRETTY_FUNCTION__ << std::endl;
}




int main() {

using FLOAT = float;

hi<int>();
hi<float>();
hi<FLOAT>();
return 0;
}
