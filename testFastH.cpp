#include "FastH.h"
#include<iostream>


void print(DynH const & h) {
  std::cout << h.type << " " << h.nbin << std::endl;
  unsigned int a[102];
  h.copy(a);
 int k=0;
  for (int i=0;i<10; ++i) {
    for (int j=0;j<10; ++j)
      std::cout << a[k++] << " ";
    std::cout << std::endl;
  }
  std::cout << a[100] << " "  << a[101] <<std::endl;
    std::cout << std::endl;
  }


int main() {

  DynH h(100,-1.,1.);
  print(h);

  for (int i=-10100; i<10100;i++)
    h.fill(float(i)/10000.f);
  print(h);

  for (int i=-1000; i<1000;i++)
    h.fill(0.32);
  print(h);


  return 0;
};
