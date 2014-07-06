#include<iostream>
int main() {

  int a = 95345;
  
  std::cout << a/4096 << " " << (a>>12) << std::endl;
  std::cout << -a/4096 << " " << ((-a)>>12) << std::endl;

   

}
