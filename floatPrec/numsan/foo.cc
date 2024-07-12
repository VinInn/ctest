#include<cmath>
#include<tuple>
#include<vector>

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


//   T b= 0.5*200, c=0.000015;
  constexpr double  b=-0.5*1.786737601482363, c=2.054360090947453e-8;
  constexpr float  bf=-0.5*1.786737601482363, cf=2.054360090947453e-8;


void  foo(double x, std::vector<double> & v) {
   auto s1 = quadSolverNaive(x,b,c);
   v.push_back(std::get<1>(s1));
}

void sfoo(double x, std::vector<double> & v) {
 auto s1 = quadSolverOpt(x,b,c);
  v.push_back(std::get<1>(s1));
}


void  foo(float x, std::vector<float> & v) {
   auto s1 = quadSolverNaive(x,bf,cf);
   v.push_back(std::get<1>(s1));
}

void sfoo(float x, std::vector<float> & v) {
 auto s1 = quadSolverOpt(x,bf,cf);
  v.push_back(std::get<1>(s1));
}


