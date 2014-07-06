struct F{
float f;
   F(float i) : f(i){}
};
struct G{
float f;
   G(float i) : f(i){}
};


void f(F);
void f(G);
void f(double);

template<typename T>
void g(T);

template<>
void g<F>(F);


float a;

void bar() {
  f(a);
  g(a);
}

