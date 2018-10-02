#include<iostream>
#include<cstdint>
#include<memory>
#include<algorithm>



int main() {

  float a[3][1000];
  float b[1000][3];

  std::cout << "a[3][1000] " << &a[0][0] - &a[0][1] << ' ' << &a[0][0] - &a[1][0] << std::endl;
  std::cout << "b[1000][3] " << &b[0][0] - &b[0][1] << ' ' << &b[0][0] - &b[1][0] << std::endl;


  return 0;

}
