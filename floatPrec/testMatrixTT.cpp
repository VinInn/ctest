#include<cmath>
#include<random>

#include "Matrix.h"
#include"TwoFloat.h"

#ifdef ALL_T
#define NOP_T
#define FLOAT_T
#define FLOAT2_T
#define DOUBLE_T
#define DOUBLE2_T
#endif

// generate matrices
template <typename M, typename Eng>
void genMatrix(M& m, Eng & eng) {
  // using T = typename std::remove_reference<decltype(m(0, 0))>::type;
  int n = M::kRows;
  std::uniform_real_distribution<float> rgen(0., 1.);

  // generate first diagonal elemets
  for (int i = 0; i < n; ++i) {
    float maxVal = i * 1.e10 / (n - 1) + 1;  // max condition is 10^10
    m(i, i) = maxVal * rgen(eng) + 1.e-9;
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      float v = 0.3f * std::sqrt( toSingle(m(i, i) * m(j, j)) );  // this makes the matrix pos defined
      m(i, j) = v * rgen(eng) + 1.e-9;;
      // m(j, i) = m(i, j);
    }
  }
}


#include<iostream>

int main() {

  int maxIter = 100000;
  
  using FF = TwoFloat<float>;
  using DD = TwoFloat<double>;

#ifdef NOP_T
{
  float maxOn=0;
  float maxOff=0;
  MatrixSym<float,5> m1,m2,m3;
  std::mt19937 eng;

for (int kk=0; kk<maxIter; ++kk) {
  genMatrix(m1, eng);
  m3 = m1;
  int n = 5;
  for (int i=0; i<n; ++i)
    maxOn = std::max(maxOn,std::abs(m3(i,i)-m1(i,i))/std::abs(m1(i,i)));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      maxOff = std::max(maxOff,std::abs(m3(i,j)-m1(i,j))/std::abs(m1(i,j)));
    }
  }
}
  std::cout << maxOn << ' ' << maxOff << std::endl;
}
#endif

#ifdef FLOAT_T
{
  float maxOn=0;
  float maxOff=0;
  MatrixSym<float,5> m1,m2,m3;
  std::mt19937 eng;

for (int kk=0; kk<maxIter; ++kk) {
  genMatrix(m1, eng);
  invert55(m1,m2);
  invert55(m2,m3);
//  invert55(m3,m2);
//  invert55(m2,m3);
  int n = 5;
  for (int i=0; i<n; ++i)
    maxOn = std::max(maxOn,std::abs(m3(i,i)-m1(i,i))/std::abs(m1(i,i)));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      maxOff = std::max(maxOff,std::abs(m3(i,j)-m1(i,j))/std::abs(m1(i,j)));
    }
  }
}
  std::cout << maxOn << ' ' << maxOff << std::endl;
}
#endif

#ifdef DOUBLE_T
{
  double maxOn=0;
  double maxOff=0;
  MatrixSym<double,5> m1,m2,m3;
  std::mt19937 eng;

for (int kk=0; kk<maxIter; ++kk) {
  genMatrix(m1, eng);
  invert55(m1,m2);
  invert55(m2,m3);
//  invert55(m3,m2);
//  invert55(m2,m3);
  int n = 5;
  for (int i=0; i<n; ++i)
    maxOn = std::max(maxOn,std::abs(m3(i,i)-m1(i,i))/std::abs(m1(i,i)));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      maxOff = std::max(maxOff,std::abs(m3(i,j)-m1(i,j))/std::abs(m1(i,j)));
    }
  }
}
  std::cout << maxOn << ' ' << maxOff << std::endl;
}
#endif

#ifdef FLOAT2_T
{
  float maxOn=0;
  float maxOff=0;
  MatrixSym<FF,5> m1,m2,m3;
  std::mt19937 eng;

for (int kk=0; kk<maxIter; ++kk) {
  genMatrix(m1, eng);
  invert55(m1,m2);
  invert55(m2,m3);
//  invert55(m3,m2);
//  invert55(m2,m3);
  int n = 5;
  for (int i=0; i<n; ++i)
    maxOn = std::max(maxOn,std::abs((m3(i,i)-m1(i,i)).hi())/std::abs(m1(i,i).hi()));
  for (int i = 0; i < n; ++i) {
//    if (kk==0) std::cout << m1(i,i).hi() << ',' << m1(i,i).lo() << std::endl;
//    if (kk==0) std::cout << m2(i,i).hi() << ',' << m2(i,i).lo() << std::endl;
//    if (kk==0) std::cout << m3(i,i).hi() << ',' << m3(i,i).lo() << std::endl;
    for (int j = 0; j < i; ++j) {
      maxOff = std::max(maxOff,std::abs( ((m3(i,j)-m1(i,j))/m1(i,j)).hi() ));
    }
  }
}
  std::cout << maxOn << ' ' << maxOff << std::endl;
}
#endif

#ifdef DOUBLE2_T
{
  double maxOn=0;
  double maxOff=0;
  MatrixSym<DD,5> m1,m2,m3;
  std::mt19937 eng;

for (int kk=0; kk<maxIter; ++kk) {
  genMatrix(m1, eng);
  invert55(m1,m2);
  invert55(m2,m3);
//  invert55(m3,m2);
//  invert55(m2,m3);
  int n = 5;
  for (int i=0; i<n; ++i)
    maxOn = std::max(maxOn,std::abs((m3(i,i)-m1(i,i)).hi())/std::abs(m1(i,i).hi()));
  for (int i = 0; i < n; ++i) {
//    if (kk==0) std::cout << m1(i,i).hi() << ',' << m1(i,i).lo() << std::endl;
//    if (kk==0) std::cout << m2(i,i).hi() << ',' << m2(i,i).lo() << std::endl;
//    if (kk==0) std::cout << m3(i,i).hi() << ',' << m3(i,i).lo() << std::endl;
    for (int j = 0; j < i; ++j) {
      maxOff = std::max(maxOff,std::abs( ((m3(i,j)-m1(i,j))/m1(i,j)).hi() ));
    }
  }
}
  std::cout << maxOn << ' ' << maxOff << std::endl;
}
#endif


  return 0;
}
