#include<iostream>
#include<cmath>


int main() {

   using  T = float;

   auto x = std::exp(T(1.)) -T(1.);
   std::cout<< std::hexfloat << x <<' '<<  std::defaultfloat << x << std::endl;

   for (int i=1; i<=25; ++i) {
      x = T(i)*x - T(1.);
     std::cout<< std::hexfloat << x <<' '<<  std::defaultfloat << x << std::endl;
   } 

   return 0;

}




