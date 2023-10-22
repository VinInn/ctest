// compile with c++ -Wall backtrace.cpp -std=c++23 -O3 -lstdc++exp -g
#include <iostream>
#include <stacktrace>

void print_stacktrace() {
    std::cout << "|\n"<< std::stacktrace::current() << "\n|" << std::endl;;
}

void b() {
     print_stacktrace();
}

void a() {
  print_stacktrace();
  b();
  print_stacktrace();
}

int main() {


  a();

  return 0;

}
