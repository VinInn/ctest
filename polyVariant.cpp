#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>


struct Z {
  virtual ~Z(){}
  virtual char operator()() const = 0;
};

struct A : public Z {
  ~A() override {}
  char operator()() const override { return 'A';}
};

struct B : public Z {
  ~B() override {}
  char operator()() const override { return 'B';}
};

struct C : public Z {
  ~C() override {}
  char operator()() const override { return 'C';}
};

using ABC = std::variant<A,B,C>;

Z &  toZ(ABC & v) { return *std::visit([](auto&& arg) -> Z* { return (Z*)(&arg);},v);}


using Cont = std::vector<ABC>;

int main() {

  Cont c;
  c.emplace_back(A());
  c.emplace_back(B());
  c.emplace_back(C());

  for ( auto & v : c) std::cout << toZ(v)() << std::endl;

  return 0;
}
