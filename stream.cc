#include<iostream>


int i=0;
void foo() { i=1;}


int main() {

  std::cout << (foo(),"switch on debug") << std::endl;
  std::cout << i << std::endl;

}
