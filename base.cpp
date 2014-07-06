#include <typeinfo>
#include <iostream>


   struct A {
     ~A() { }
   int a;
   };

   struct B : public A {
     virtual ~B() { }
   int b;
   };

   struct C : public B {
     virtual ~C() { }
    int c;
   };


int main()   {
    B * b = new C;
    A * a = b;
    void * p = dynamic_cast<void*>(b);

	std::cout << b << " " << p << ", " << a << " " << &(b->a) << std::endl;

    delete b; // c' leack?


    return 0;
  }
