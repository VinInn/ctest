#include<cmath>
// #define sinl sin
#include<type_traits>
#include<cstdio>
int main() {
 printf("%a %a\n", sinf(1000000.f),float(sin(1000000.))); 
  return 0;
}

/*
float  expd(float x) {
   return ::exp(x);
}

float  expd2(float x) {
   return ::exp(double(x));
}


float  sind(float x) {
   return sinl(x);
}

float p2(float x) {
  return x+3.15;
}


float  sind2(float x) {
   return sin(double(x));
}



double foo(double x);
float foof(float x);
float foov();

template<typename T>
T fun0( T(*f)(T), T x) {
        return f(x);
}

*/
/*
template<typename F>
auto fun1(F, typename std::result_of<F>::type x) -> decltype(x) {
	return F(x);
}
 
template<typename F>
auto fun2(F, decltype(F(0)) x) -> decltype(x) {
        return F(x);
}
*/

/*
float  f0(float x) {
   return fun0(foo,double(x));
}
*/

/*
float  f1(float x) {
   return fun1(foo,x);
}

float  f2(float x) {
   return fun2(foo,double(x));
}

void bha() {
  decltype(foo()) q = 3;
  decltype(foo(0)) a = 3;
  std::result_of<foof>::type = a;
}
*/
