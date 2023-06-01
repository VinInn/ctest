#include "Xoshiro.h"


#include<iostream>

int main() {

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
   auto x = g.next();
   for (int i=0; i<4; i++) std::cout << x[i] << ' ';
   x = g.next();
   for (int i=0; i<4; i++) std::cout << x[i] << ' ';
   std::cout << std::endl;
 }

/*
 {
  Xoshiro<XoshiroType::TwoSums,uint64_t> g(0);
   for (int i=0; i<8; i++) std::cout << g() << ' ';
   std::cout << std::endl;
 }
*/

}
