#include<algorithm>
#include<cstdlib>
#include<iostream>
#include<cmath>


void a() {

  std::cout << abs(-3.4) << std::endl; 
  std::cout << ::abs(-3.4) << std::endl;
  std::cout << std::abs(-3.4) << std::endl;

}


void b() {

  using namespace std;

  std::cout << abs(-3.4) << std::endl;
  std::cout << ::abs(-3.4) << std::endl;
  std::cout << std::abs(-3.4) << std::endl;

}


int main() {

  a();
  std::cout << std::endl;
  b();

   return 0;

}
