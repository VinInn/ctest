#include <iostream>
#include <cstring>

int main() {

unsigned int u, u1,u2;
int i1,i2;

   i1=-4072;
   u1 = i1;

   i2= 4070;
   u2 = i2;

   int k; memcpy(&k,&u1,4);
   std::cout << i1 << ' ' << u1 << ' '<< k << std::endl;

   u = u1-u2;
   memcpy(&k,&u,4);
   std::cout << i1-i2 << ' ' << u1-u2 << ' '<< k << std::endl;

   u = u2-u1;
   memcpy(&k,&u,4);
   std::cout << i2-i1 << ' ' << u2-u1 << ' '<< k << std::endl;

   u = u2+u1;
   memcpy(&k,&u,4);
   std::cout << i2+i1 << ' ' << u2+u1 << ' '<< k << std::endl;
   

}
