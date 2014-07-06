#include<cstdio>
#include<cstdlib>
#include<iostream>
#include<limits>
#include<cassert>
#include<cmath>

#include "approx_atan2.h"

// performance test
#include <x86intrin.h>
inline unsigned long long rdtsc() {
 unsigned int i = 0;
 return __rdtscp(&i);
}

template <typename T> 
inline T toPhi (T phi) { 
  T result = phi;
  while (result > T(M_PI)) result -= T(2*M_PI);
  while (result <= -T(M_PI)) result += T(2*M_PI);
  return result;
}


template <typename T> 
inline T deltaPhi (T phi1, T phi2) { 
  T result = phi1 - phi2;
  while (result > T(M_PI)) result -= T(2*M_PI);
  while (result <= -T(M_PI)) result += T(2*M_PI);
  return result;
}


inline bool phiLess(float x, float y) {
  auto ix = phi2int(toPhi(x));
  auto iy = phi2int(toPhi(y));

  return (ix-iy)<0;

}

int main() {

  constexpr long long maxint = (long long)(std::numeric_limits<int>::max())+1LL;
  constexpr int pi2 =  int(maxint/2LL);
  constexpr int pi4 =  int(maxint/4LL);
  constexpr int pi34 = int(3LL*maxint/4LL);

  std::cout << "pi,  pi2,  pi4, p34 " << maxint << ' ' << pi2 << ' ' << pi4 << ' ' << pi34 << ' ' << pi2+pi4  << '\n';
  std::cout << "Maximum value for int: " << std::numeric_limits<int>::max() << '\n';
  std::cout << "Maximum value for int+2: " << std::numeric_limits<int>::max()+2 << '\n';
  std::cout << "Maximum value for int+1 as LL: " << (long long)(std::numeric_limits<int>::max())+1LL << std::endl;

  std::cout << "Maximum value for short: " << std::numeric_limits<short>::max() << '\n';
  std::cout << "Maximum value for short+2: " << short(std::numeric_limits<short>::max()+short(2)) << '\n';
  std::cout << "Maximum value for short+1 as int: " << (int)(std::numeric_limits<short>::max())+1 << std::endl;


  auto d = float(M_PI) -std::nextafter(float(M_PI),0.f);
  std::cout << "abs res at pi for float " << d << ' ' << phi2int(d) << std::endl;
  std::cout << "abs res at for int " << int2dphi(1) << std::endl;
  std::cout << "abs res at for short " << short2phi(1) << std::endl;




  assert(-std::numeric_limits<int>::max() == (std::numeric_limits<int>::max()+2));

  assert(phiLess(0.f,2.f));
  assert(phiLess(6.f,0.f));
  assert(phiLess(3.2f,0.f));
  assert(phiLess(3.0f,3.2f));

  assert(phiLess(-0.3f,0.f));
  assert(phiLess(-0.3f,0.1f));
  assert(phiLess(-3.0f,0.f));
  assert(phiLess(3.0f,-3.0f));
  assert(phiLess(0.f,-3.4f));

  // go around the clock
  float phi1= -7.;
  while (phi1<8) {
    auto p1 = toPhi(phi1);
    auto ip1 = phi2int(p1);
    std::cout << "phi1 " << phi1 << ' ' << p1 << ' ' << ip1 << ' ' << int2phi(ip1) << std::endl;

    float phi2= -7.2;
    while (phi2<8) {
    auto p2 = toPhi(phi2);
    auto ip2 = phi2int(p2);
    std::cout << "phi2 " << phi2 << ' ' <<  deltaPhi(phi1,phi2)  << ' ' <<  deltaPhi(phi2,phi1)
	      << ' ' << int2phi(ip1-ip2) << ' ' << int2phi(ip2-ip1)   
	      << ' ' <<  toPhi(phi2+phi1) << ' ' << int2phi(ip1+ip2) << std::endl;
      phi2+=1;
    }

    phi1+=1;
  }



  return 0;

}
