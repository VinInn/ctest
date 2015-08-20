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
void go() {
  std::cout <<' '<< std::endl;
  T a=1., b=-0.5*1.786737601482363, c=2.054360090947453e-8;
  auto s1 = quadSolverNaive(a,b,c);
  std::cout << "Naive Solution "<<  std::scientific << std::setprecision(std::numeric_limits<T>::digits10+3) << std::get<0>(s1) << ' ' << std::get<1>(s1) << std::endl;
  auto s2 = quadSolverOpt(a,b,c);
  std::cout <<  "Opt  Solution "<<  std::scientific << std::setprecision(std::numeric_limits<T>::digits10+3) << std::get<0>(s2) << ' ' << std::get<1>(s2) << std::endl;
  std::cout << std::endl;
}

int main(){


  go<float>();
  go<double>();
  // go<__float128>();

  return 0;

}
