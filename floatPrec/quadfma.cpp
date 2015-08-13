#include<iostream>
#include<cmath>
#include<limits>

template<typename T>
void print(T x) {
 std::cout<< std::hexfloat << double(x) <<' '<<  std::defaultfloat << double(x) << std::endl;
}

double comp(double a, double b, double c) { 
  return ((double(0.25)*b*b)-a*c);
}


template<typename T> 
void go(T x){
  T a =  x + 94906265.625;
  T c =  94906268.375 -x;
  T b =  x -189812534;

  print ((T(0.25)*b*b)-a*c);
  print (fmal(-a,T(4)*c,b*b));
  print (comp(a,b,c));
}

int main(int argc, char **) {
  double k = argc-1;
  go<float>(k);
  go<double>(k);
  go<__float128>(k);
  return 0;
}
