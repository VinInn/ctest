#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>


int main(){

  std::cout
    << 0xf<< ','       
    << 0xff<< ','   
    << 0xfff<< ','
    << 0xffff<< ','
    << 0xfffff<< ','
    << 0xffffff
    << std::endl;  
  
  int d = 244;
  unsigned char cd(d);
  char * p = (char *)(&cd);
  std::cout << d/4 << std::endl;
  std::cout << std::hex << (int)(cd) << std::endl;
  std::cout << std::hex << (int)(unsigned char)(*p) << std::endl;
  
  // std::cout << "\xf \xff \xfff \xffff \xfffff \xffffff\n";
  
  int c;
  sscanf("7B","%x",&c);
  std::cout << " " << (char) c << std::endl;
  
  union FourByte {
    int i;
    unsigned char c[4];
  };
  
  FourByte r1;  r1.i=::rand();
  FourByte r2;  r2.i=::rand();
  
  std::cout  << std::hex << (int)r1.c[0] << (int)r1.c[1]<<(int)r1.c[2]<< (int)r1.c[3] << std::endl;
  std::cout << std::hex << ::rand() << ',' << ::rand() << " " << RAND_MAX<< std::endl;

  int z1 = 0xfff;
  int z2 = 0xf00;
  int ma = 0xf0;

  std::cout << std::hex << z1 << " " << z2 << std::endl;
  
  z1 &= ~ma;
  z2 &= ~ma;

  std::cout << std::hex << ma << " " << ( ~ma) << std::endl;
  std::cout << std::hex << z1 << " " << z2 << std::endl;
  
  return 0;
}
