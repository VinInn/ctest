#include<cmath>
#include<random>
#include <cassert>


#include "Matrix.h"
#include"TwoFloat.h"
#include<iostream>

#ifdef ALL_T
#define NOP_T
#define FLOAT_T
#define FLOAT2_T
#define DOUBLE_T
#define DOUBLE2_T
#endif

#define VERIFY

template <typename M>
inline void verify(M const& m) {
#ifdef VERIFY
#warning "Verify ON"
  int n = M::kRows;
  for (int i = 0; i < n; ++i)
       assert(toSingle(m(i,i))>0);
  //check minors
  auto d = toSingle(m(0, 0)*m(1,1)) - toSingle(m(0, 1)*m(1,0));
  if (d<0) std::cout << "??? " << d << std::endl;
  assert(d > -1.e-8);
  auto d3 = toSingle(m(1, 0)*m(2,1)) - toSingle(m(2, 0)*m(1,1));
  auto d2 = toSingle(m(1, 0)*m(2,2)) - toSingle(m(2, 0)*m(1,2));
  auto d1 = toSingle(m(1, 1)*m(2,2)) - toSingle(m(1, 2)*m(2,1));
  auto dd = toSingle(m(0,0))*d1-toSingle(m(0,1))*d2+toSingle(m(0,2))*d3;
  if (dd<0) std::cout << "??? " << dd << std::endl;
  assert(dd > -1.e-8);
}
#endif 

// generate matrices
template <typename M, typename Eng>
void genMatrix(M& m, Eng & eng) {
  // using T = typename std::remove_reference<decltype(m(0, 0))>::type;
  int n = M::kRows;
  std::uniform_real_distribution<float> rgen(0., 1.);

  // generate first diagonal elemets
  for (int i = 0; i < n; ++i) {
    float maxVal = i * 1.e5 / (n - 1) + 1;  // max condition is 10^5 as  min-generated is 10^-9
    m(i, i) = maxVal * rgen(eng) + 1.e-10;
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      float v = 0.3f * std::sqrt( toSingle(m(i, i) * m(j, j)) );  // this makes the matrix pos defined
      m(i, j) = v * rgen(eng) + 1.e-10;
      if (rgen(eng)<0.5f) m(i, j) = -m(i, j);
      // m(j, i) = m(i, j);
    }
  }
}


#include<iostream>

template<typename T,typename TT=T>
void go(int maxIter) {
  T maxOn=0;
  T maxOff=0;
  MatrixSym<TT,5> m1,m2,m3;
  std::mt19937 eng;

for (int kk=0; kk<maxIter; ++kk) {
  genMatrix(m1, eng);
  verify(m1);
  invert55(m1,m2);
  verify(m2);
  invert55(m2,m3);
  verify(m3);
//  invert55(m3,m2);
//  invert55(m2,m3);
  int n = 5;
  for (int i=0; i<n; ++i) {
    maxOn = std::max(maxOn,std::abs(toSingle(  (m3(i,i)-m1(i,i))/m1(i,i) )));
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      maxOff = std::max(maxOff,std::abs(toSingle( (m3(i,j)-m1(i,j))/m1(i,j) )));
    }
  }
}
  std::cout << maxOn << ' ' << maxOff << std::endl;
}


int main() {

  int maxIter = 1000000;
  
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
  verify(m1);
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
  go<float>(maxIter);
#endif

#ifdef DOUBLE_T
  go<double>(maxIter);
#endif

#ifdef FLOAT2_T
  go<float,FF>(maxIter);
#endif

#ifdef DOUBLE2_T
  go<double,DD>(maxIter);
#endif


  return 0;
}
