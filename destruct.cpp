struct Vah {int i; int j;};

auto foo() {  return Vah{1,2};}

int bar() {
  auto const [a,b]=foo();
  return a;

}
#include<iostream>
int main() {
  std::cout << bar() << std::endl;
}
