// clang++ -fsanitize=numerical -Wall -g ~innocent/public/ctest/floatPrec/quadSolvers.cpp -fno-sanitize-trap=all -fsanitize-recover=all -Ofast -march=native
#include<cmath>
#include<tuple>
#include<limits>
#include<quadmath.h>


template<typename T>
inline T det(T a, T b, T c) {
  // compute determinant for equation ax^2 + 2bx + c= 0
  return std::sqrt(b*b-a*c);
}

template<>
inline __float128 det(__float128 a,  __float128 b, __float128 c) {
  // compute determinant for equation ax^2 + 2bx + c= 0
  return ::sqrtq(b*b-a*c);
}


template<typename T>
inline std::tuple<T,T> quadSolverNaive(T a, T b, T c) {
  // solve equation ax^2 + 2bx + c= 0
  // using naive solution (as at college)
  auto d = -T(1)/a;
  auto q = det(a,b,c);
  return std::make_tuple(d*(b-q),d*(b+q));  
}


template<typename T>
inline std::tuple<T,T> quadSolverOpt(T a, T b, T c) {
  // solve equation ax^2 + 2bx + c= 0
  // using stable algorithm
  auto q = -(std::copysign(det(a,b,c),b)+b);
  return std::make_tuple(q/a,c/q);
}

template<>
inline std::tuple<__float128,__float128> quadSolverOpt(__float128 a, __float128 b, __float128 c) {
  // solve equation ax^2 + 2bx + c= 0
  // using stable algorithm
  auto q = -(::copysignq(det(a,b,c),b)+b);
  return std::make_tuple(q/a,c/q);
}




#include<iostream>
#include<iomanip>

using LD = long double;

template<typename T> 
void print(T x) {
  std::cout<< std::hexfloat << x <<' ' <<  std::scientific << std::setprecision(std::numeric_limits<T>::digits10+3) << LD(x) << std::endl;
}


template<typename T>
void go(T a) {
  std::cout <<' '<< std::endl;
//   T b= 0.5*200, c=0.000015;
  T b=-0.5*1.786737601482363, c=2.054360090947453e-8;
  auto s1 = quadSolverNaive(a,b,c);
  auto s2 = quadSolverOpt(a,b,c);
  if (s1==s2) std::cout << "precise!!!" << std::endl;
  std::cout << "Naive Solution "<<  std::scientific << std::setprecision(std::numeric_limits<T>::digits10+3) << LD(std::get<0>(s1)) << ' ' << LD(std::get<1>(s1)) << std::endl;
  std::cout << " Opt  Solution "<<  std::scientific << std::setprecision(std::numeric_limits<T>::digits10+3) << LD(std::get<0>(s2)) << ' ' << LD(std::get<1>(s2)) << std::endl;
  std::cout << std::endl;
}


template<typename T>
void circle(int q) {
  std::cout <<' '<< std::endl;

  constexpr T micron = 1.e-3;
  constexpr T one = 1000.;
  constexpr T halfChord = one/2;

  
  T x1 = std::sqrt(q>1 ? 66. : 77.);
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

int main(int argc, char **){

  go<__float128>(argc);
  go<double>(argc);
  go<float>(argc);

  circle<double>(argc);
  circle<float>(argc);

  
  return 0;

}
