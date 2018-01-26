#include<iostream>

   using w32 = unsigned int;
   using w64 = unsigned long long;


void print(const w32 * w) {
   std::cout << w[0] << ' ' << w[1] << std::endl;
}  


int main() {

   using w32 = unsigned int;
   using w64 = unsigned long long;


   w32 a[2] = {5,1025};

   w64 w = w64(a[0])<<32 | a[1];
   w64 r = w64(a[1])<<32 | a[0];

   print (a);
   print ((const w32 *)(&w));
   print ((const w32 *)(&r));

   return 0;
}

