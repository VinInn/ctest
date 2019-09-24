#include<iostream>


struct A {

 A() { std::cout << "A"<< std::endl;}
 ~A() { std::cout << "~A"<< std::endl;}
 double a;

};



struct B {

 B() { std::cout << "B"<< std::endl;}
 ~B() { std::cout << "~B"<< std::endl;}
 float a,b;

};


union U {
  U(){}
  ~U(){}
  A a;
  B b;

};



int main() {

 
 {
  U u;
 }

 {
  U u;
  std::cout << "= a" << std::endl;
  u.a = A();
  std::cout << "= b" << std::endl;
  u.b = B();
 }
  std::cout << "end" << std::endl;


  return 0;

}

