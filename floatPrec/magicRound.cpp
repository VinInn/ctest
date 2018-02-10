#include <cstdint>
#include <cmath>
#include <limits>
#include <cstring>
#include <quadmath.h>

constexpr float roundF = 1.5*std::pow(2,23);//  (digits-1)
constexpr double roundD = 1.5*std::pow(2,52);
constexpr __float128 roundL = 1.5q*powl(2.q,112.q);

template<typename T>
struct rtr {};

template<>
struct rtr<float> {
  using SInt = short;
  static constexpr float round = roundF; 
};

template<>
struct rtr<double> {
  using SInt = int;
  static constexpr double round = roundD; 
};

template<>
struct rtr<__float128> {
  using SInt = long long;
  static constexpr double round = roundL; 
};

template<typename T>
auto toInt(T t) -> typename rtr<T>::SInt {typename rtr<T>::SInt i; memcpy(&i,&t, sizeof(i)); return i;}


#include<iostream>

template<typename T>
void go() {

  
  std::cout << std::numeric_limits<T>::digits << std::endl;

  std::cout << rtr<T>::round << std::endl;

  T a = T(-12.345)+rtr<T>::round;
  
  std::cout << toInt(a) << std::endl;
  std::cout << double(a-rtr<T>::round) << std::endl;

  
  std::cout << std::endl;

}
  

int main() {

  go<float>();
  go<double>();
  go<__float128>();

  return 0;

}
