#include <iostream>

void f(double) { std::cout << "right\n"; }
template <class T> struct X { void g() { f(1); } };
void f(int) { std::cout << "wrong\n"; }

int main() { X<int>().g(); }
