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

// #define VERIFY

template <typename M>
inline bool verify(M const& m, bool vv=true) {
  bool ret=true;
#ifdef VERIFY
#warning "Verify ON"
  int n = M::kRows;
  for (int i = 0; i < n; ++i) {
     auto d = toSingle(m(i,i));
     if (vv && d<0) std::cout << "??? on " << i << ' ' << d << std::endl;
//     assert(d>-1.e-8);
     if (d<1.e-8) ret=false;
  }
  //check minors
  for (int i = 0; i < n-1; ++i) {
    auto d = toSingle(m(i+0, i+0)*m(i+1,i+1)) - toSingle(m(i+0, i+1)*m(i+1,i+0));
    if (vv && d<0) std::cout << "??? m2 " << i << ' ' << d << std::endl;
//    assert(d > -1.e-8);
    if (d<1.e-8) ret=false;
    if (i>0) continue;;
    auto d3 = toSingle(m(i+1, i+0)*m(i+2,i+1) - m(i+2, 0)*m(i+1,i+1));
    auto d2 = toSingle(m(i+1, i+0)*m(i+2,i+2) - m(i+2, 0)*m(i+1,i+2));
    auto d1 = toSingle(m(i+1, i+1)*m(i+2,i+2) - m(i+1, 2)*m(i+2,i+1));
    auto dd = toSingle(m(i+0,i+0)*d1-m(i+0,i+1)*d2+m(i+0,i+2)*d3);
    if (vv && dd<0) std::cout << "??? m3 " << i << ' ' << dd << std::endl;
//    assert(dd > -1.e-8);
    if (d<1.e-8) ret=false;
  }
#endif 
   return ret;
}

// generate matrices
template <typename M, typename Eng>
void genMatrix(M& m, Eng & eng) {
  // using T = typename std::remove_reference<decltype(m(0, 0))>::type;
  int n = M::kRows;
  std::uniform_real_distribution<float> rgen(0., 1.);
  do {
  // generate first diagonal elemets
  for (int i = 0; i < n; ++i) {
    float maxVal = i * 1.e5 / (n - 1) + 1;  // max condition is 10^5 as  min-generated is 10^-9
    m(i, i) = maxVal * (rgen(eng) + 1.e-10);
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      float v = 0.3f * std::sqrt( toSingle(m(i, i) * m(j, j)) );  // this makes the matrix pos defined
      m(i, j) = v * (rgen(eng) + 1.e-10);
      if (rgen(eng)<0.5f) m(i, j) = -m(i, j);
      // m(j, i) = m(i, j);
    }
  }
  } while(!verify(m,false));
}

#include <typeinfo>
#include<iostream>

template<typename T,typename TT=T>
void go(int maxIter) {
  std::cout << "testing " << typeid(TT).name() << std::endl;
  T maxOn=0;
  T maxOff=0;
  MatrixSym<TT,5> m1,m2,m3;
  std::mt19937 eng;

for (int kk=0; kk<maxIter; ++kk) {
  bool v = true;
  genMatrix(m1, eng);
  v &= verify(m1);
  invert55(m1,m2);
  v &= verify(m2);
  invert55(m2,m3);
  v &= verify(m3);
//  invert55(m3,m2);
//  invert55(m2,m3);
//  if (!v) continue;
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

  int maxIter = 5000000;
  
  using FF = TwoFloat<float>;
  using DD = TwoFloat<double>;

#ifdef NOP_T
{
  std::cout << "testing NOP" << std::endl;
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
