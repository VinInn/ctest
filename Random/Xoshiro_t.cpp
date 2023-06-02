#include "Xoshiro.h"


#include<iostream>

int main() {

#ifdef __AVX2__
 std::cout << "avx2 supported" << std::endl;
#endif

  std::cout << "vector size " << XoshiroSS::vector_size << std::endl;

 {
   XoshiroP g(3);
   for (int i=0; i<8; i++) std::cout << g() << ' '; 
   std::cout << std::endl;
 }
 {
   XoshiroPP g(3);
   for (int i=0; i<8; i++) std::cout << g() << ' ';
   std::cout << std::endl;
 }
 {
   XoshiroSS g(3);
   for (int i=0; i<8; i++) std::cout << g() << ' ';
   std::cout << std::endl;
 }
 {
   XoshiroSS g(0);
   for (int i=0; i<8; i++) std::cout << g() << ' ';
   std::cout << std::endl;
 }
 {
   XoshiroSS g(0);
   auto n = XoshiroSS::vector_size;
   auto x = g.next();
   for (int i=0; i<n; i++) std::cout << x[i] << ' ';
   x = g.next();
   for (int i=0; i<n; i++) std::cout << x[i] << ' ';
   std::cout << std::endl;
 }
 
 {
  Xoshiro<XoshiroType::TwoMuls,uint64_t> g(0);
   for (int i=0; i<8; i++) std::cout << g() << ' ';
   std::cout << std::endl;
 }

 {
  Xoshiro<XoshiroType::TwoSums,uint64_t> g(0);
   for (int i=0; i<8; i++) std::cout << g() << ' ';
   std::cout << std::endl;
 }


}
