constexpr int foo(int a, int b) { if (a<b) a+=b;  return a; }


constexpr int a = foo(2, 3);
