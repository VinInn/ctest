#include <immintrin.h>

#include "x64Time.h"

#include <iostream>
#include <cmath>

namespace atest {
  double get(int);
  double func(double);
  void sink(double);
}


inline double f1(double x, double y, double t) {

  return x* std::cos(t) + y*std::sin(t);

}

inline void zero() {
#ifdef ZERO
   _mm256_zeroupper();
#endif
}

inline void go (int i) {
#define auto double

//   zero();
 
   using namespace atest;
  
   auto x = get(i);
   auto y = get(i);
   auto t = get(i);

  //zero();

  __m256d w = _mm256_set1_pd(t);
  w = _mm256_add_pd(w,w);
  w = _mm256_hadd_pd(w,w);  
 double q[4];
 _mm256_storeu_pd(q,w);
 t = q[2];
	
  zero();
  auto f = f1(x,y,t);

 __m256d a = _mm256_set1_pd(f);
 
  zero();
  f = func(f);

  // zero();

  __m256d b = _mm256_add_pd(a,_mm256_set1_pd(f));

  double s[4];
  _mm256_storeu_pd(s,b);
  zero();
  sink(s[0]);

}

int main() {

  std::cout << "hi" << std::endl;

  auto s = rdtsc();
  go(0);
  auto e1 = rdtsc()-s;

  s = rdtsc();
  for (int i=0; i!=10; i++)
    go(i);
  auto e2 = rdtsc()-s;

  std::cout << "ticks " << e1 << std::endl;
  std::cout << "ticks " << e2 << std::endl;

  return 0;
}
