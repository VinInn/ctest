#include<iostream>


int main() {

  alignas(128) double d=0;
  alignas(128) double v[4]={0,};

  using aligned_block alignas(128) = float[4];
  aligned_block al;

  using aldouble alignas(128) = double;
  aldouble d2=0;
  // aldouble v2[4]={0,};

  struct alignas(128) AlDouble { double x;};
  AlDouble d3={0};
  AlDouble v3[4]={0,};


  std::cout << sizeof(d) << " " << &d << std::endl;
  std::cout <<  long(&v[4]) - long(&v[0]) << " " << v   << std::endl;
  std::cout << sizeof(d2) << " " << &d2 << std::endl;
  //std::cout <<  long(&v2[4]) - long(&v2[0]) << " " << v2   << std::endl;
  std::cout << sizeof(d3) << " " << &d3 << std::endl;
  std::cout <<  long(&v3[4]) - long(&v3[0]) << " " << v3   << std::endl;


  std::cout << sizeof(aligned_block) << " " << &al << std::endl;


  return 0;
};
