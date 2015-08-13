#include<iostream>
#include<cmath>


template<typename T> 
void print(T x) {
 std::cout<< std::hexfloat << x <<' '<<  std::defaultfloat << x << std::endl;
}

template<typename F>
void go() {

  constexpr F micron = 1.e-3;
  constexpr F one = 1000.;
  print(micron);
  print(one+micron);
  print(one-micron);
  F a = one+ micron;
  F b = one- micron;
  print(a-b);

  constexpr F pi = std::acos(-1.);
  constexpr F pi4 = 0.25*pi;
  print(pi);

  auto xa = cos(pi4)*a;
  auto ya = sin(pi4)*a;
  print(xa);
  print(ya);
  auto da = xa*xa+ya*ya;
  auto a2= a*a;
  print(da);
  print(a2);
  print(da-a2); 
  print(std::sqrt(da));
  print(std::sqrt(a2));

  auto xb = cos(pi4)*b;
  auto yb = sin(pi4)*b; 
  print(xb);
  print(yb);


  constexpr F halfChord = 500.;
  for (F sagita = F(10.); sagita>micron; sagita*=F(0.5)) {
    std::cout << "sagita ";print(sagita);
    auto radius = (sagita*sagita+halfChord*halfChord)/(F(2)*sagita);
    std::cout << "radius ";print(radius);
    auto x0 = radius+sagita;
    std::cout << "x0 "; print(x0);
    std::cout << "x0-r "; print(x0-radius);    
    std::cout << "x0^2-r^2 "; print(x0*x0-radius*radius);
    std::cout << "r^2-h^2 "; print(radius*radius-halfChord*halfChord);   
    std::cout << "recomputed sagita "; print(radius-std::sqrt(radius*radius-halfChord*halfChord));
    std::cout << "recomputed sagita 2"; print(halfChord*halfChord/(radius+std::sqrt(radius*radius-halfChord*halfChord)));

  }


}


int main() {

  go<double>();
  std::cout << std::endl;
  go<float>();

  return 0;
}
