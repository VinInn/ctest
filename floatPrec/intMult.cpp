#include<iostream>


int main() {

   int a = (1<<30) + 1;

   std::cout << a << ' ' << 2*a << ' ' << 3*a << ' ' << 4*a << std::endl; 
   std::cout << a << ' ' << 5*a << ' ' << 6*a << ' ' << 7*a << std::endl;

   int b =a;
   for (int i=0; i<7; ++i) std::cout << (b+=a) << ' ';
   std::cout << std::endl;

   return 0;
}
