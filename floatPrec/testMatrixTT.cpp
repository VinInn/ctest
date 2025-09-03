#include<cmath>
#include<random>

#include "Matrix.h"

// generate matrices
template <typename M, typename Eng>
void genMatrix(M& m, Eng & eng) {
  // using T = typename std::remove_reference<decltype(m(0, 0))>::type;
  int n = M::kRows;
  std::uniform_real_distribution<float> rgen(0., 1.);

  // generate first diagonal elemets
  for (int i = 0; i < n; ++i) {
    float maxVal = i * 1.e12 / (n - 1) + 1;  // max condition is 10^12
    m(i, i) = maxVal * rgen(eng);
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      float v = 0.3f * std::sqrt(float(m(i, i) * m(j, j)));  // this makes the matrix pos defined
      m(i, j) = v * rgen(eng);
      // m(j, i) = m(i, j);
    }
  }
}


#include"TwoFloat.h"

#include<iostream>

int main() {

  using FF = TwoFloat<float>;
  using DD = TwoFloat<double>;

{
  float maxOn=0;
  float maxOff=0;
  MatrixSym<float,5> m1,m2,m3;
  std::mt19937 eng;

for (int kk=0; kk<1000; ++kk) {
  genMatrix(m1, eng);
  invert55(m1,m2);
  invert55(m2,m3);
  invert55(m3,m2);
  invert55(m2,m3);
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

{
  double maxOn=0;
  double maxOff=0;
  MatrixSym<double,5> m1,m2,m3;
  std::mt19937 eng;

for (int kk=0; kk<1000; ++kk) {
  genMatrix(m1, eng);
  invert55(m1,m2);
  invert55(m2,m3);
  invert55(m3,m2);
  invert55(m2,m3);
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



{
  float maxOn=0;
  float maxOff=0;
  MatrixSym<FF,5> m1,m2,m3;
  std::mt19937 eng;

for (int kk=0; kk<1000; ++kk) {
  genMatrix(m1, eng);
  invert55(m1,m2);
  invert55(m2,m3);
  invert55(m3,m2);
  invert55(m2,m3);
  int n = 5;
  for (int i=0; i<n; ++i)
    maxOn = std::max(maxOn,std::abs((m3(i,i)-m1(i,i)).hi())/std::abs(m1(i,i).hi()));
  for (int i = 0; i < n; ++i) {
    if (kk==0) std::cout << m1(i,i).hi() << ',' << m1(i,i).lo() << std::endl;
    if (kk==0) std::cout << m2(i,i).hi() << ',' << m2(i,i).lo() << std::endl;
    if (kk==0) std::cout << m3(i,i).hi() << ',' << m3(i,i).lo() << std::endl;
    for (int j = 0; j < i; ++j) {
      maxOff = std::max(maxOff,std::abs( ((m3(i,j)-m1(i,j))/m1(i,j)).hi() ));
    }
  }
}
  std::cout << maxOn << ' ' << maxOff << std::endl;
}

{
  double maxOn=0;
  double maxOff=0;
  MatrixSym<DD,5> m1,m2,m3;
  std::mt19937 eng;

for (int kk=0; kk<1000; ++kk) {
  genMatrix(m1, eng);
  invert55(m1,m2);
  invert55(m2,m3);
  invert55(m3,m2);
  invert55(m2,m3);
  int n = 5;
  for (int i=0; i<n; ++i)
    maxOn = std::max(maxOn,std::abs((m3(i,i)-m1(i,i)).hi())/std::abs(m1(i,i).hi()));
  for (int i = 0; i < n; ++i) {
    if (kk==0) std::cout << m1(i,i).hi() << ',' << m1(i,i).lo() << std::endl;
    if (kk==0) std::cout << m2(i,i).hi() << ',' << m2(i,i).lo() << std::endl;
    if (kk==0) std::cout << m3(i,i).hi() << ',' << m3(i,i).lo() << std::endl;
    for (int j = 0; j < i; ++j) {
      maxOff = std::max(maxOff,std::abs( ((m3(i,j)-m1(i,j))/m1(i,j)).hi() ));
    }
  }
}
  std::cout << maxOn << ' ' << maxOff << std::endl;
}



  return 0;
}
