#include<iostream>
#include<memory>
#include<cstdlib>

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


  std::cout << sizeof(d) << " " << (long(&d)&127) << std::endl;
  std::cout <<  long(&v[4]) - long(&v[0]) << " " << (long(v)&127)   << std::endl;
  std::cout << sizeof(d2) << " " << (long(&d2)&127) << std::endl;
  //std::cout <<  long(&v2[4]) - long(&v2[0]) << " " << (long(v2)&127)   << std::endl;
  std::cout << sizeof(d3) << " " << (long(&d3)&127) << std::endl;
  std::cout <<  long(&v3[4]) - long(&v3[0]) << " " << (long(v3)&127)   << std::endl;


  std::cout << sizeof(aligned_block) << " " << alignof(aligned_block)<< ' ' << (long(&al)&127) << std::endl;

  auto k = new float;
  std::cout << (long(k)&127) << std::endl;
  k = new float;
  std::cout << (long(k)&127) << std::endl;

  auto p = new aligned_block;
  std::cout << (long(p)&127) << std::endl;
  p = std::aligned_alloc(alignof(aligned_block),sizeof(aligned_block)));
  std::cout << (long(p)&127) << std::endl;



  return 0;
};
