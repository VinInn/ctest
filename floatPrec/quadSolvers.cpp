#include<cmath>
#include<tuple>
#include<limits>

template<typename T>
inline T det(T a, T b, T c) {
  // compute determinant for equation ax^2 + 2bx + c= 0
  return std::sqrt(b*b-a*c);
}

template<typename T>
inline std::tuple<T,T> quadSolverNaive(T a, T b, T c) {
  // solve equation ax^2 + 2bx + c= 0
  // using naive solution (as at college)
  auto d = -T(1)/a;
  return std::make_tuple(d*(b+det(a,b,c)),d*(b-det(a,b,c)));  
}


template<typename T>
inline std::tuple<T,T> quadSolverOpt(T a, T b, T c) {
  // solve equation ax^2 + 2bx + c= 0
  // using stable algorithm
  auto q = -(std::copysign(det(a,b,c),b)+b);
  return std::make_tuple(q/a,c/q);
}




#include<iostream>
#include<iomanip>


template<typename T> 
void print(T x) {
  std::cout<< std::hexfloat << x <<' ' <<  std::scientific << std::setprecision(std::numeric_limits<T>::digits10+3) << x << std::endl;
}


template<typename T>
void go() {
  std::cout <<' '<< std::endl;
  T a=1., b=-0.5*1.786737601482363, c=2.054360090947453e-8;
  auto s1 = quadSolverNaive(a,b,c);
  std::cout << "Naive Solution "<<  std::scientific << std::setprecision(std::numeric_limits<T>::digits10+3) << std::get<0>(s1) << ' ' << std::get<1>(s1) << std::endl;
  auto s2 = quadSolverOpt(a,b,c);
  std::cout <<  "Opt  Solution "<<  std::scientific << std::setprecision(std::numeric_limits<T>::digits10+3) << std::get<0>(s2) << ' ' << std::get<1>(s2) << std::endl;
  std::cout << std::endl;
}


template<typename T>
void circle() {
  std::cout <<' '<< std::endl;

  constexpr T micron = 1.e-3;
  constexpr T one = 1000.;
  constexpr T halfChord = one/2;

  
  T x1 = std::sqrt(77.);
  std::cout << "x1 "; print(x1);

  for (auto sagita = T(10.); sagita>micron; sagita*=T(0.5)) {
    std::cout << "sagita ";print(sagita);
    auto radius = (sagita*sagita+halfChord*halfChord)/(T(2)*sagita);
    std::cout << "radius ";print(radius);
    auto x0 = x1 + radius - sagita;
    auto xm = x1-sagita;

    std::cout << "x0 "; print(x0);
    std::cout << "xm "; print(xm);
    std::cout << "x0-r "; print(x0-radius);    
    std::cout << "x0^2-r^2 "; print(x0*x0-radius*radius);
    std::cout << "r^2-h^2 "; print(radius*radius-halfChord*halfChord);

    print(x0-std::sqrt(radius*radius-halfChord*halfChord));
    print(x1-(x0-std::sqrt(radius*radius-halfChord*halfChord)));
    auto s1 = quadSolverOpt(T(1),-radius, halfChord*halfChord);
    print(xm+std::get<1>(s1));
    print(x1-(xm+std::get<1>(s1)));
    
    std::cout << std::endl;
  }
  
    std::cout << std::endl;
}

int main(){


  go<float>();
  go<double>();
  // go<__float128>();

  circle<float>();
  circle<double>();

  
  return 0;

}
