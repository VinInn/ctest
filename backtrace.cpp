// compile with c++ -Wall backtrace.cpp -std=c++23 -O3 -lstdc++exp -g
#include <iostream>
#include <string>
#include <stacktrace>

inline
void print_stacktrace() {
   std::string trace;
   for (auto & entry : std::stacktrace::current() ) trace += entry.description() + '#';

    std::cout << '|' << trace << '|' << std::endl;;
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
