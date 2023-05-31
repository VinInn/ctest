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
}
