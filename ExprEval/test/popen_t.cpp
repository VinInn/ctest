#include "popenCPP.h"
#include<iostream>



int main() {

  try { 
    auto ss = popenCPP("c++ -v");
    char c;
    while (ss->get(c)) std::cout << c;
    std::cout << std::endl;

    auto s1 = popenCPP("uuidgen | sed 's/-//g'");
    std::string n1;
    while (s1->get(c)) n1+=c;
    std::cout << n1 << std::endl;

  }catch(...) {
    std::cout << "error...." << std::endl;
  }

  return 0;

}
