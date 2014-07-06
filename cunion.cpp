   union fx {
     constexpr fx(int i): x(i){}
     constexpr float f() const { return a;}
     constexpr float i() const { return x;}
     float a;
     int x;
   };

float go(float x) {

  constexpr float a = fx(3).f();
  constexpr int  b = fx(3).i();

  return a*x +b;

}
