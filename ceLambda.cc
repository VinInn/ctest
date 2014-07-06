struct Bar {  constexpr Bar(float i):f(i){}; float f;};
float foo1(float x) {
   constexpr Bar z{0};

   auto f = [=](auto a, auto b) -> Bar { return z;};

   return f(x,x).f;

}

float foo2(float x) {
   const Bar z{0};

   auto f = [=](auto a, auto b) -> Bar { return z;};

   return f(x,x).f;

}

float foo3(float x) {
   constexpr Bar z{0};

   auto f = [=](float a, float b) -> Bar { return z;};

   return f(x,x).f;

}

