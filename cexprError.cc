constexpr int foo0(bool c) {
  return c ? 1 : throw "only true is valid";
} 

int bar();

constexpr int foo1(bool c) {
  return c ? 1 : bar();
} 


int compile() {
  constexpr int a0 = foo0(true);
  constexpr int a1 = foo1(true);
  return a0+a1;
}

#ifndef DOC
int compileNot() {
  constexpr int a0 = foo0(false);
  constexpr int a1 = foo1(false);
  return a0+a1;
}
#endif
